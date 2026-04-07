# main

> Source: `nyutemplate/main.tex`
> Master Note: [[erc3_full_documentation]]

# Introduction

## Introduction

- Your introduction goes here!
- Use `itemize` to organize your main points.

\vskip 1cm

\begin{block}{Examples}
Some examples of commonly used commands and features are included, to help you get started.
\end{block}

# Some \LaTeX{} Examples

## Tables and Figures

## Tables and Figures

- Use `tabular` for basic tables --- see Table~\ref{tab:widgets}, for example.
- You can upload a figure (JPEG, PNG or PDF) using the files menu.
- To include it in your document, use the `includegraphics` command (see the comment below in the source code).

\begin{table}
```text
Item & Quantity \\\hline
Widgets & 42 \\
Gadgets & 13
```
\caption{\label{tab:widgets}An example table.}
\end{table}

## Mathematics

## Readable Mathematics

Let $X_1, X_2, \ldots, X_n$ be a sequence of independent and identically distributed random variables with $\text{E}[X_i] = \mu$ and $\text{Var}[X_i] = \sigma^2 < \infty$, and let
$$S_n = \frac{X_1 + X_2 + \cdots + X_n}{n}
= \frac{1}{n}\sum_{i}^{n} X_i$$
denote their mean. Then as $n$ approaches infinity, the random variables $\sqrt{n}(S_n - \mu)$ converge in distribution to a normal $\mathcal{N}(0, \sigma^2)$.
