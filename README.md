# jieba-next

[![PyPI version](https://badge.fury.io/py/jieba-next.svg)](https://badge.fury.io/py/jieba-next)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jieba-next.svg)](https://pypi.org/project/jieba-next/)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mxcoras/jieba-next/publish-pypi.yml?branch=main&style=flat-square)](https://github.com/mxcoras/jieba-next/actions)
[![PyPI - Downloads](https://img.shields.io/pypi/dd/jieba-next.svg)](https://pypistats.org/packages/jieba-next)
[![License](https://img.shields.io/pypi/l/jieba-next.svg)](https://opensource.org/licenses/MIT)

`jieba-next` 是 [jieba_fast](https://github.com/deepcs233/jieba_fast) 的一个现代化分支，旨在提供对 Python 3.9+ 的支持，并利用 Cython 进行了代码优化和加速。

`jieba_fast` 本身是经典中文分词库 `jieba` 的一个 CPython 加速版本。本项目在 `jieba_fast` 的基础上，更新了构建系统，并用 Cython 重新实现了部分核心算法，解决了[内存泄漏问题](https://github.com/deepcs233/jieba_fast/issues/26)，并提升了可维护性。

## 项目特点

- **现代化**：支持 Python 3.9 及更高版本，不再支持 Python 2。
- **性能**：利用 Cython 重新实现了生成 DAG（有向无环图）及计算最优路径的算法，以提升分词速度。
- **兼容性**：力求与原版 `jieba` 和 `jieba_fast` 的分词结果保持一致。
- **易于安装**：使用现代化的构建工具，提供多平台的预编译二进制包（wheels），简化安装过程。
- **易于使用**：可以作为 `jieba` 的直接替代品，只需 `import jieba_next as jieba`。

## 当前状态

本项目目前处于早期开发阶段：

- 已完成基础功能测试，可以正确执行分词任务。
- 与原 `jieba_fast` 仓库的分词结果具有一致性。
- 性能方面略低于原 `jieba_fast` 仓库，但仍远强于原版 `jieba`，后续将持续进行优化。
- 测试覆盖尚不完整，欢迎贡献测试用例。

## 安装

您可以通过 pip 从 PyPI 安装：

```bash
pip install jieba-next
```

或者从源码安装：

```bash
git clone https://github.com/mxcoras/jieba-next.git
cd jieba-next
pip install .
```

## 使用示例

可以像使用 `jieba` 或 `jieba_fast` 一样使用 `jieba-next`。

```python
import jieba_next as jieba

text = "在输出层后再增加CRF层，加强了文本间信息的相关性，针对序列标注问题，每个句子的每个词都有一个标注结果，对句子中第i个词进行高维特征的抽取，通过学习特征到标注结果的映射，可以得到特征到任意标签的概率，通过这些概率，得到最优序列结果"

print("-".join(jieba.lcut(text, HMM=True)))
print('-'.join(jieba.lcut(text, HMM=False)))

```

输出:

```text
在-输出-层后-再-增加-CRF-层-，-加强-了-文本-间-信息-的-相关性-，-针对-序列-标注-问题-，-每个-句子-的-每个-词-都-有-一个-标注-结果-，-对-句子-中-第-i-个-词-进行-高维-特征-的-抽取-，-通过-学习-特征-到-标注-结果-的-映射-，-可以-得到-特征-到-任意-标签-的-概率-，-通过-这些-概率-，-得到-最优-序列-结果
```

```text
在-输出-层-后-再-增加-CRF-层-，-加强-了-文本-间-信息-的-相关性-，-针对-序列-标注-问题-，-每个-句子-的-每个-词-都-有-一个-标注-结果-，-对-句子-中-第-i-个-词-进行-高维-特征-的-抽取-，-通过-学习-特征-到-标注-结果-的-映射-，-可以-得到-特征-到-任意-标签-的-概率-，-通过-这些-概率-，-得到-最优-序列-结果
```

## 算法

- 基于前缀词典实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图 (DAG)。
- 采用动态规划查找最大概率路径, 找出基于词频的最大切分组合。
- 对于未登录词，采用了基于汉字成词能力的 HMM 模型，并使用了 Viterbi 算法。

## 鸣谢

"结巴"中文分词原作者: SunJunyi  
jieba_fast 仓库作者: deepcs233
