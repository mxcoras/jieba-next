# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from cpython.long cimport PyLong_AsLong, PyLong_FromLong
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem, PyDict_Contains
from cpython.list cimport PyList_New, PyList_Append, PyList_Size
from cpython.tuple cimport PyTuple_New, PyTuple_SetItem, PyTuple_GetItem
from cpython.float cimport PyFloat_AsDouble, PyFloat_FromDouble
from cpython.object cimport PyObject
from cpython.sequence cimport PySequence_Size, PySequence_GetSlice, PySequence_GetItem
from libc.math cimport log, INFINITY
from libc.stdlib cimport malloc, free

cdef double MIN_FLOAT = -3.14e100

def _get_DAG_and_calc(dict FREQ, str sentence, list route, double total):
    cdef Py_ssize_t N = len(sentence)
    cdef Py_ssize_t i, k, idx, max_x, t_list_len, x
    cdef double logtotal = log(total)
    cdef double max_freq
    cdef double fq_2, fq_last
    cdef long fq

    cdef Py_ssize_t* DAG_c = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * 20 * N)
    cdef Py_ssize_t* points = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * N)
    cdef double* _route_c = <double*> malloc(sizeof(double) * 2 * (N + 1))

    if not DAG_c or not points or not _route_c:
        if DAG_c: free(DAG_c)
        if points: free(points)
        if _route_c: free(_route_c)
        raise MemoryError("Failed to allocate memory")

    _route_c[N * 2] = 0.0
    _route_c[N * 2 + 1] = 0.0

    for i in range(N):
        points[i] = 0

    cdef object frag
    cdef object val
    for k in range(N):
        i = k
        while i < N and points[k] < 12:
            frag = sentence[k:i + 1]
            val = FREQ.get(frag, None)
            if val is None:
                break
            if val:
                DAG_c[k * 20 + points[k]] = i
                points[k] += 1
            i += 1
        if points[k] == 0:
            DAG_c[k * 20] = k
            points[k] = 1

    for idx in range(N - 1, -1, -1):
        max_freq = MIN_FLOAT
        max_x = 0
        t_list_len = points[idx]
        for i in range(t_list_len):
            x = DAG_c[idx * 20 + i]
            fq = FREQ.get(sentence[idx:x + 1], 1)
            if fq == 0:
                fq = 1

            fq_2 = _route_c[(x + 1) * 2]
            fq_last = log(fq) - logtotal + fq_2

            if fq_last >= max_freq:
                max_freq = fq_last
                max_x = x

        _route_c[idx * 2] = max_freq
        _route_c[idx * 2 + 1] = <double> max_x

    for i in range(N + 1):
        route.append(int(_route_c[i * 2 + 1]))

    free(DAG_c)
    free(points)
    free(_route_c)


def _viterbi(str obs, str _states, dict start_p, dict trans_p, dict emit_p):
    cdef Py_ssize_t obs_len = len(obs)
    cdef int states_num = len(_states)

    if obs_len == 0 or states_num == 0:
        return (0.0, [])

    cdef double* V = <double*> malloc(sizeof(double) * obs_len * states_num)
    cdef int* path_prev = <int*> malloc(sizeof(int) * obs_len * states_num)
    if not V or not path_prev:
        if V: free(V)
        if path_prev: free(path_prev)
        raise MemoryError("Failed to allocate memory")

    cdef dict PrevStatus = {'B': 'ES', 'M': 'MB', 'S': 'SE', 'E': 'BM'}
    cdef int* prev0 = <int*> malloc(sizeof(int) * states_num)
    cdef int* prev1 = <int*> malloc(sizeof(int) * states_num)
    if not prev0 or not prev1:
        if V: free(V)
        if path_prev: free(path_prev)
        if prev0: free(prev0)
        if prev1: free(prev1)
        raise MemoryError("Failed to allocate memory")

    cdef int j, i, best_idx, cand_idx
    cdef double em_p, max_prob, prob
    cdef str y, y0, prevs

    cdef list states_list = [None] * states_num
    cdef list emit_dicts = [None] * states_num
    cdef list trans_dicts = [None] * states_num
    for j in range(states_num):
        y = _states[j]
        states_list[j] = y
        emit_dicts[j] = emit_p.get(y, {})
        trans_dicts[j] = trans_p.get(y, {})

    for j in range(states_num):
        y = states_list[j]
        prevs = PrevStatus[y]
        prev0[j] = _states.find(prevs[0])
        prev1[j] = _states.find(prevs[1])

    for j in range(states_num):
        y = states_list[j]
        em_p = (<dict> emit_dicts[j]).get(obs[0], MIN_FLOAT)
        V[0 * states_num + j] = start_p[y] + em_p
        path_prev[0 * states_num + j] = j

    cdef int pidx
    for i in range(1, obs_len):
        for j in range(states_num):
            y = states_list[j]
            em_p = (<dict> emit_dicts[j]).get(obs[i], MIN_FLOAT)

            max_prob = MIN_FLOAT
            best_idx = -1

            cand_idx = prev0[j]
            if cand_idx >= 0:
                y0 = states_list[cand_idx]
                prob = V[(i - 1) * states_num + cand_idx] \
                       + (<dict> trans_dicts[cand_idx]).get(y, MIN_FLOAT) \
                       + em_p
                if prob > max_prob:
                    max_prob = prob
                    best_idx = cand_idx

            cand_idx = prev1[j]
            if cand_idx >= 0:
                y0 = states_list[cand_idx]
                prob = V[(i - 1) * states_num + cand_idx] \
                       + (<dict> trans_dicts[cand_idx]).get(y, MIN_FLOAT) \
                       + em_p
                if prob > max_prob:
                    max_prob = prob
                    best_idx = cand_idx

            if best_idx < 0:
                prevs = PrevStatus[y]
                y0 = prevs[0] if prevs[0] >= prevs[1] else prevs[1]
                best_idx = _states.find(y0)
                if best_idx < 0:
                    best_idx = j

            V[i * states_num + j] = max_prob
            path_prev[i * states_num + j] = best_idx

    cdef int idxE = _states.find('E')
    cdef int idxS = _states.find('S')
    cdef double max_last = V[(obs_len - 1) * states_num + idxE]
    cdef int now_idx = idxE
    if V[(obs_len - 1) * states_num + idxS] > max_last:
        max_last = V[(obs_len - 1) * states_num + idxS]
        now_idx = idxS

    cdef list final_path = [''] * obs_len
    for i in range(obs_len - 1, -1, -1):
        final_path[i] = states_list[now_idx]
        now_idx = path_prev[i * states_num + now_idx]

    free(V)
    free(path_prev)
    free(prev0)
    free(prev1)
    return (max_last, final_path)
