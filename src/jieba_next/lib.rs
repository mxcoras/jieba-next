use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use std::collections::HashMap;
use std::sync::OnceLock;

const MIN_FLOAT: f64 = -3.14e100_f64;

fn char_boundaries(s: &str) -> Vec<usize> {
	// Return byte indices for each char start plus the end: len = chars + 1
	let mut idxs: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
	idxs.push(s.len());
	idxs
}

// High-performance prefix dictionary (Trie)
#[derive(Default)]
struct Node {
	children: HashMap<char, usize>,
	freq: i64, // 0 means not a word (prefix only)
}

#[pyclass]
struct PrefixDict {
	nodes: Vec<Node>,
	log_total: f64,
}

impl PrefixDict {
	fn new_empty(log_total: f64) -> Self {
		let mut nodes = Vec::with_capacity(1024);
		nodes.push(Node { children: HashMap::new(), freq: 0 }); // root
		Self { nodes, log_total }
	}

	fn insert_word(&mut self, word: &str, freq: i64) {
		let mut cur = 0usize;
		for ch in word.chars() {
			let next = match self.nodes[cur].children.get(&ch) {
				Some(&idx) => idx,
				None => {
					let idx = self.nodes.len();
					self.nodes[cur].children.insert(ch, idx);
					self.nodes.push(Node { children: HashMap::new(), freq: 0 });
					idx
				}
			};
			cur = next;
		}
		// Set frequency for full word (can be zero too, kept for fidelity)
		self.nodes[cur].freq = freq;
	}

	fn scan_from(&self, sentence: &str, bounds: &[usize], start_idx: usize) -> Vec<(usize, i64)> {
		// Return up to 12 ends with their frequencies (>0), or fallback (start_idx, 1)
		let mut res: Vec<(usize, i64)> = Vec::with_capacity(12);
		let mut cur = 0usize;
		for i in start_idx..(bounds.len() - 1) {
			// walk next char
			let ch = sentence[bounds[i]..bounds[i + 1]].chars().next().unwrap();
			match self.nodes[cur].children.get(&ch) {
				Some(&nxt) => {
					cur = nxt;
					let fq = self.nodes[cur].freq;
					if fq > 0 {
						res.push((i, fq));
						if res.len() >= 12 { break; }
					}
				}
				None => break,
			}
		}
		if res.is_empty() {
			// fallback: single char with freq treated as 1
			res.push((start_idx, 1));
		}
		res
	}
}

#[pymethods]
impl PrefixDict {
	#[new]
	fn py_new(freq: Bound<'_, PyDict>, total: f64) -> PyResult<Self> {
		let log_total = total.ln();
		let mut me = PrefixDict::new_empty(log_total);
		for (k, v) in freq.iter() {
			let key: String = k.extract()?;
			let fv: i64 = v.extract().unwrap_or(0);
			me.insert_word(&key, fv);
		}
		Ok(me)
	}

	/// Compute route (size N+1) using the internal trie for fast DAG+DP.
	fn get_dag_and_calc(&self, sentence: &str) -> PyResult<Vec<isize>> {
		let n_chars = sentence.chars().count();
		let mut route_next: Vec<usize> = vec![0; n_chars + 1];
		if n_chars == 0 {
			return Ok(vec![0]);
		}
		let bounds = char_boundaries(sentence);

		// Build DAG with freqs
		let mut dag: Vec<Vec<(usize, i64)>> = Vec::with_capacity(n_chars);
		for k in 0..n_chars {
			dag.push(self.scan_from(sentence, &bounds, k));
		}

		// DP
		let mut route_score = vec![0f64; n_chars + 1];
		for idx in (0..n_chars).rev() {
			let mut max_val = MIN_FLOAT;
			let mut max_x = idx;
			for &(x, fq) in &dag[idx] {
				let fq_val = if fq == 0 { 1 } else { fq } as f64;
				let cand = fq_val.ln() - self.log_total + route_score[x + 1];
				if cand >= max_val {
					max_val = cand;
					max_x = x;
				}
			}
			route_score[idx] = max_val;
			route_next[idx] = max_x;
		}

		let mut out: Vec<isize> = Vec::with_capacity(n_chars + 1);
		for i in 0..=n_chars {
			out.push(route_next[i] as isize);
		}
		Ok(out)
	}
}

#[pyfunction(name = "_get_DAG_and_calc")]
fn _get_dag_and_calc(
	freq: Bound<'_, PyDict>,
	sentence: &str,
	route: Bound<'_, PyList>,
	total: f64,
) -> PyResult<()> {
	let n_chars = sentence.chars().count();
	if n_chars == 0 {
		// Still append one element like Cython does (N+1)
		route.call_method1("append", (0_isize,))?;
		return Ok(());
	}

	let bounds = char_boundaries(sentence);

	// DAG: for each k -> list of end positions (inclusive)
	// Match Cython behavior: stop when prefix missing; collect only entries with non-zero freq
	let mut dag: Vec<Vec<usize>> = Vec::with_capacity(n_chars);
	for k in 0..n_chars {
		let mut ends: Vec<usize> = Vec::with_capacity(12);
		let mut i = k;
		while i < n_chars && ends.len() < 12 {
			let frag = &sentence[bounds[k]..bounds[i + 1]];
			// None -> break; Some(0) -> continue (no push)
			match freq.get_item(frag)? {
				None => break,
				Some(val) => {
					// Truthiness check like Cython: if val
					// In jieba dict, values are ints; treat 0 as false, others as true
					let is_true = if let Ok(iv) = val.extract::<i64>() {
						iv != 0
					} else if let Ok(fv) = val.extract::<f64>() {
						fv != 0.0
					} else {
						val.is_truthy()?
					};
					if is_true {
						ends.push(i);
					}
				}
			}
			i += 1;
		}
		if ends.is_empty() {
			ends.push(k);
		}
		dag.push(ends);
	}

	// DP route arrays
	let mut route_score = vec![0f64; n_chars + 1];
	let mut route_next = vec![0usize; n_chars + 1];
	let logtotal = total.ln();

	for idx in (0..n_chars).rev() {
		let mut max_freq = MIN_FLOAT;
		let mut max_x = idx; // default to self
		for &x in &dag[idx] {
			let frag = &sentence[bounds[idx]..bounds[x + 1]];
			// fq = FREQ.get(..., 1); if fq == 0 -> 1
			let mut fq = match freq.get_item(frag)? { Some(v) => v.extract::<i64>().unwrap_or(1_i64), None => 1_i64 };
			if fq == 0 {
				fq = 1;
			}
			let fq_last = (fq as f64).ln() - logtotal + route_score[x + 1];
			if fq_last >= max_freq {
				max_freq = fq_last;
				max_x = x;
			}
		}
		route_score[idx] = max_freq;
		route_next[idx] = max_x;
	}

	// Append N+1 entries like Cython
	for i in 0..=n_chars {
		route.call_method1("append", (route_next[i] as isize,))?;
	}
	Ok(())
}

#[pyfunction]
fn _viterbi(
	obs: &str,
	states: &str,
	start_p: Bound<'_, PyDict>,
	trans_p: Bound<'_, PyDict>,
	emit_p: Bound<'_, PyDict>,
) -> PyResult<(f64, Vec<String>)> {
	// Use cached HMM model to avoid PyDict overhead on every step
	static HMM_CACHE: OnceLock<HmmModel> = OnceLock::new();

	let model: &HmmModel = if let Some(m) = HMM_CACHE.get() {
		m
	} else {
		let built = HmmModel::from_py(states, start_p, trans_p, emit_p)?;
		let _ = HMM_CACHE.set(built);
		// Safe to unwrap because either we just set it or another thread did
		HMM_CACHE.get().unwrap()
	};

	let obs_len = obs.chars().count();
	let states_num = model.states_chars.len();
	if obs_len == 0 || states_num == 0 {
		return Ok((0.0, Vec::new()));
	}

	// PrevStatus mapping indices
	let mut prev0 = vec![None::<usize>; states_num];
	let mut prev1 = vec![None::<usize>; states_num];
	for j in 0..states_num {
		let y = model.states_chars[j];
		let (p0, p1) = prev_status(y);
		prev0[j] = model.state_index.get(&p0).copied();
		prev1[j] = model.state_index.get(&p1).copied();
	}

	// Prepare obs as chars
	let obs_chars: Vec<char> = obs.chars().collect();

	// DP tables
	let mut v = vec![MIN_FLOAT; obs_len * states_num];
	let mut path_prev = vec![0usize; obs_len * states_num];

	// Initialize i=0
	for j in 0..states_num {
		let em_p = model.emit[j].get(&obs_chars[0]).copied().unwrap_or(MIN_FLOAT);
		let sp = model.start[j];
		v[j] = sp + em_p;
		path_prev[j] = j;
	}

	for i in 1..obs_len {
		for j in 0..states_num {
			let em_p = model.emit[j].get(&obs_chars[i]).copied().unwrap_or(MIN_FLOAT);
			let mut max_prob = MIN_FLOAT;
			let mut best_idx = None::<usize>;

			if let Some(ci) = prev0[j] {
				let prob = v[(i - 1) * states_num + ci] + model.trans[ci][j] + em_p;
				if prob > max_prob { max_prob = prob; best_idx = Some(ci); }
			}
			if let Some(ci) = prev1[j] {
				let prob = v[(i - 1) * states_num + ci] + model.trans[ci][j] + em_p;
				if prob > max_prob { max_prob = prob; best_idx = Some(ci); }
			}

			let bi = if let Some(b) = best_idx {
				b
			} else {
				// Fallback mirrors Cython: choose max(prevs[0], prevs[1]) by char order
				let (p0, p1) = prev_status(model.states_chars[j]);
				let y0 = if p0 >= p1 { p0 } else { p1 };
				model.state_index.get(&y0).copied().unwrap_or(j)
			};
			v[i * states_num + j] = max_prob;
			path_prev[i * states_num + j] = bi;
		}
	}

	// Choose last state between 'E' and 'S'
	let idx_e = model.state_index.get(&'E').copied().unwrap_or(0);
	let idx_s = model.state_index.get(&'S').copied().unwrap_or(0);
	let mut max_last = v[(obs_len - 1) * states_num + idx_e];
	let mut now_idx = idx_e;
	let vs = v[(obs_len - 1) * states_num + idx_s];
	if vs > max_last { max_last = vs; now_idx = idx_s; }

	// Backtrack
	let mut final_path: Vec<String> = vec![String::new(); obs_len];
	for i in (0..obs_len).rev() {
		final_path[i] = model.states_chars[now_idx].to_string();
		now_idx = path_prev[i * states_num + now_idx];
	}

	Ok((max_last, final_path))
}

#[pymodule]
fn jieba_next_rust(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<PrefixDict>()?;
	m.add_function(wrap_pyfunction!(_get_dag_and_calc, &m)?)?;
	m.add_function(wrap_pyfunction!(_viterbi, &m)?)?;
	Ok(())
}

// ---------------- HMM Model cache -----------------

struct HmmModel {
	states_chars: Vec<char>,
	state_index: HashMap<char, usize>,
	start: Vec<f64>,                 // [states]
	trans: Vec<Vec<f64>>,            // [states][states]
	emit: Vec<HashMap<char, f64>>,   // per state emission probs keyed by char
}

impl HmmModel {
	fn from_py(
		states: &str,
		start_p: Bound<'_, PyDict>,
		trans_p: Bound<'_, PyDict>,
		emit_p: Bound<'_, PyDict>,
	) -> PyResult<Self> {
		let states_chars: Vec<char> = states.chars().collect();
		let n = states_chars.len();
		let mut state_index: HashMap<char, usize> = HashMap::with_capacity(n);
		for (i, &c) in states_chars.iter().enumerate() { state_index.insert(c, i); }

		// start
		let mut start = vec![MIN_FLOAT; n];
		for (i, &c) in states_chars.iter().enumerate() {
			let key = c.to_string();
			start[i] = match start_p.get_item(&key)? { Some(v) => v.extract::<f64>().unwrap_or(MIN_FLOAT), None => MIN_FLOAT };
		}

		// trans
		let mut trans = vec![vec![MIN_FLOAT; n]; n];
		for (pi, &pc) in states_chars.iter().enumerate() {
			let pk = pc.to_string();
			if let Some(any) = trans_p.get_item(&pk)? {
				let row = any.downcast::<PyDict>()?;
				for (ni, &nc) in states_chars.iter().enumerate() {
					let nk = nc.to_string();
					trans[pi][ni] = match row.get_item(&nk)? { Some(v) => v.extract::<f64>().unwrap_or(MIN_FLOAT), None => MIN_FLOAT };
				}
			}
		}

		// emit: per state map<char, f64>
		let mut emit: Vec<HashMap<char, f64>> = Vec::with_capacity(n);
		for &c in &states_chars {
			let key = c.to_string();
			let mut m: HashMap<char, f64> = HashMap::new();
			if let Some(any) = emit_p.get_item(&key)? {
				let d = any.downcast::<PyDict>()?;
				for (k, v) in d.iter() {
					// Keys are one-character strings in model; extract first char if longer
					let ks: String = k.extract()?;
					if let Some(ch) = ks.chars().next() {
						let val = v.extract::<f64>().unwrap_or(MIN_FLOAT);
						m.insert(ch, val);
					}
				}
			}
			emit.push(m);
		}

		Ok(Self { states_chars, state_index, start, trans, emit })
	}
}

#[inline]
fn prev_status(y: char) -> (char, char) {
	match y { 'B' => ('E', 'S'), 'M' => ('M', 'B'), 'S' => ('S', 'E'), 'E' => ('B', 'M'), _ => ('S', 'E') }
}

