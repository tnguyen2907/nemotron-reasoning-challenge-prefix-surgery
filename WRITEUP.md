# 309th Place - Prefix Surgery

## Context

The competition asks participants to create a rank-32-or-smaller LoRA adapter
for `NVIDIA-Nemotron-3-Nano-30B`. The hidden evaluation covers six reasoning
families, including two equation variants: numeric operation puzzles and symbol
ciphers.

- Business context: [Competition Overview](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview)
- Data context: [Competition Data](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data)

## Overview of My Approach

### The idea behind prefix surgery

This project started with a question that occurred to me while reading failed
reasoning traces:

> If most of a model's reasoning is already correct, why train it on a completely
> different solution? Why not train only from the point where it goes wrong?

Imagine that a model produces:

```text
2x + 2 = 0
2x = -2
x = 1        <- first wrong step
Therefore the answer is 1.
```

A normal correction dataset might replace the whole response. My idea was to
keep:

```text
2x + 2 = 0
2x = -2
```

and train only:

```text
x = -1
Therefore the answer is -1.
```

That led to **prefix surgery**:

1. keep the model's response exactly up to its first invalid line;
2. cut the response at that point;
3. add a corrected continuation; and
4. place the kept prefix in the prompt so completion-only loss is applied only
   to the repair.

The hope was that this would focus training on the model's mistake instead of
spending the training signal on reasoning it already knew.

### Choosing the baseline

I found the public baseline through
[@afr1ste's adapter guide](https://www.kaggle.com/code/afr1ste/nemotron-0-86-tinker-adapter-guide).
The adapter itself was released by
[@kienngx](https://www.kaggle.com/models/kienngx/nemotron-nano-30b-trained),
along with a
[submission notebook](https://www.kaggle.com/code/kienngx/nvidia-nemotron-trained-models-submission).

The Kien adapter was already very strong on most tasks in my local evaluation:

| Task | Accuracy on all 9,500 rows |
| --- | ---: |
| roman_numeral | 100.00% |
| gravity | 99.75% |
| unit_conversion | 99.75% |
| cipher_text | 99.30% |
| bit_manipulation | 78.71% |
| equation_digit_ops | 76.91% |
| **equation_symbol_cipher** | **8.02%** |

Because most categories were already close to solved, I focused only on
`equation_symbol_cipher`, the task with the largest obvious headroom.

### The experiment

I trained two adapters from the same Kien warm start:

1. **Arm A - prefix surgery.** Keep the verified part of Kien's original trace
   and train a corrected continuation.
2. **Arm B - clean rewrite.** Throw away Kien's trace and train a newly written
   solution from the beginning.

Both arms used the same target rows, replay data, optimizer, and seed. The main
difference was how the symbol-cipher trace was written.

The short version of the result is that neither approach improved the target
task. Although the easy tasks stayed near 99-100% for both arms, neither arm
learned the missing symbol-cipher search. But some interesting behaviors
appeared:

- Arm A retained much more of the model's existing `equation_digit_ops` ability.
- Arm B nearly erased its `equation_digit_ops` ability.

Somehow, Arm A actually performed slightly better on the private dataset, even
though its public score and held-out accuracy were slightly worse. Therefore,
I'm definitely not claiming that my method improved the model for this
competition, because it did not.

***This write-up is mainly about the interesting difference in behavior between
Arm A and Arm B.***

The implementation and the executed end-to-end notebook are available in
[this repository](https://github.com/tnguyen2907/nemotron-reasoning-challenge-prefix-surgery).
I used Claude Code and Codex extensively to
implement, inspect, and test the code.

## Details of the Approach

### 1. Building a solver

Each symbol-cipher puzzle hides two things:

1. what each operator means; and
2. which decimal digit each glyph represents.

For example, a puzzle might secretly use:

```text
* means multiply, then add one

` = 5
! = 6
[ = 4
{ = 9
```

The same glyph must represent the same digit everywhere in the puzzle, and two
different glyphs cannot represent the same digit.

My solver uses
[@lkevincc's public solver](https://github.com/lkevincc0/kaggle-nemotron-equation-symbolic)
and [solver discussion](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/698293)
as its foundation. I changed how the search was organized, but not its essential
coverage.

In simple terms, the solver:

1. tries possible meanings for the operators;
2. tries digit assignments that can satisfy each example;
3. removes assignments that contradict another example;
4. enforces one consistent digit mapping across the whole puzzle; and
5. applies the surviving program to the question.

A mapping that works for one equation is not enough. It must work for every
equation in the puzzle.

The final solver matched lkevincc's public result: it found a gold-matching
solution for 800 of the 823 symbol-cipher rows.

### 2. Creating the training traces

#### What Kien's original traces look like

Kien's symbol-cipher responses are unusually regular. All 823 begin with the
same opening and letter-table format, and about 97% contain the same
example-conversion and question-conversion sections.

A typical response looks like:

```text
We need to infer the transformation rule from the examples.

First, let me assign letters to each symbol:
` -> A
! -> B
[ -> C
...
Operators:
* -> x
- -> y

Converting examples to letter form:
...

Operator x: unknown
Operator y: unknown

The question operator is y, which is unknown.
As the question operator is unknown, we default to concatenation.
...
\boxed{wrong answer}
```

The explicit concatenation fallback appears on 517 of the 823 rows. This made
the failure easy to recognize: the model often performed useful setup, reached
an unknown operator, and then concatenated the inputs.

#### The shortcut I tried

The Python solver could discover both the operators and the digits, but the
digit search was the difficult part. One equation could have many possible
mappings. The solver had to combine constraints across all examples until only
a globally consistent mapping remained.

Writing that complete search into every training trace would have been long and
unnatural, and many traces would have pushed against the model's context limit.
I therefore tried a shortcut:

1. include lightweight reasoning about the operators, such as recognizing that
   four-digit outputs from two-digit inputs look multiplication-like;
2. insert the digit mapping found by the Python solver;
3. plug that mapping into every example to show that it works; and
4. use it to calculate the final answer.

For example:

```text
Structure for operator x:
result widths 4,3,4 from two-digit operands -> multiplication-like.

Narrow E1:
56 x 49 = 2744, so with +1 we get 2745.

The consistent digit assignment is:
A=5, B=6, C=4, D=7, ...

Check E1: 56 x 49 + 1 = 2745, matching the encoded result.
Check E2: 32 x 20 + 1 = 641, matching the encoded result.
...
```

This trace shows how to **use and check** a completed mapping. It does not show
how the solver searched through competing mappings and discovered it.

I hoped that, after seeing many correct examples, the model might learn that
missing digit search internally. This was the central bet of the experiment,
and it did not work.

#### Arm A: prefix surgery

Because Kien's traces followed a stable template, I could build a deterministic
trace surgeon rather than ask another language model to guess where the response
went wrong.

For each trace, the script:

1. reads the symbol-to-letter table and checks that it is consistent;
2. compares every converted example with the original puzzle;
3. recomputes any arithmetic that has already appeared;
4. keeps the last fully correct line; and
5. cuts before the first malformed or incorrect line.

The cut is different for every row. Some responses fail while building the
letter table. Others correctly convert several examples before reaching
`Operator x: unknown`.

For one example, Kien's response begins:

```text
We need to infer the transformation rule from the examples.

First, let me assign letters to each symbol:
` -> A
! -> B
[ -> C
" -> D
' -> E
{ -> J
...
Operators:
* -> x
- -> y

Converting examples to letter form:

「`!*[{」 = 「'"[」:  <- Wrong: the original output is `'"[``. Kien drops the final backtick.
  input:  ` ! * [ { -> AB x CJ
  output: " ' [ -> DEC
```

Arm A cuts before the incorrect conversion header and continues with the
corrected conversion:

```text
「`!*[{」 = 「'"[`」:
  input:  ` ! * [ { -> AB x CJ
  output: ' " [ ` -> EDCA

「\'*'>」 = 「![@」:
  input:  \ ' * ' > -> HE x EF
  output: ! [ @ -> BCG

...

Operator x: multiplication, then add one
Operator y: reversed subtraction

Structure for operator x:
result widths 4,3,4 from two-digit operands -> multiplication-like.

The consistent digit assignment is
A=5, B=6, C=4, D=7, ...

Check E1: 56 x 49 + 1 = 2745, matching the example.
...
Query: 62 - 44 = 18.
Converting back: 18 -> @&
\boxed{@&}
```

The original prefix is part of the prompt. Only the corrected conversion and
the new continuation are used as the training completion.

#### Arm B: rewriting the whole solution

Arm B receives the same solver result, but its trace starts from scratch:

```text
I will solve the symbol equation directly as a digit cipher.

The multiplication equation is the strongest anchor.
Operator * is multiply, then add one.
Operator - is reversed subtraction.

Digit assignment:
!=6, "=7, &=8, '=2, ...

Check E1: 56 x 49 + 1 = 2745.
...
Query: 62 - 44 = 18.
The encoded answer is @&.
\boxed{@&}
```

This is cleaner to read, but it does not preserve Kien's original notation or
reasoning rhythm. It also has the same missing-search problem as Arm A: the
correct digit assignment is stated rather than derived.

#### Final trace checks

I did not admit a trace just because it ended with the correct answer. Every
generated trace had to pass these checks:

- every narrated arithmetic statement recomputed correctly;
- every letter used by Arm A was defined first;
- the final boxed answer equaled the training answer;
- the complete prompt and completion fit within 8,192 tokens; and
- held-out rows were excluded from training.

The final training corpus contained 282 symbol-cipher traces per arm after the
context-length filter.

### 3. Training

It has been observed in previous public experiments that fine-tuning heavily on
the symbol-cipher task can cause regressions in other categories, as notably
discussed in
[@kemshim's post](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/703479).
To reduce forgetting, I added correct traces produced by the Kien baseline as
replay anchors.

Each arm contained:

| Training component | Rows |
| --- | ---: |
| Symbol-cipher target traces | 282 |
| Bit-manipulation anchors | 300 |
| Text-cipher anchors | 300 |
| Gravity anchors | 300 |
| Roman-numeral anchors | 300 |
| Unit-conversion anchors | 300 |
| **Total** | **1,782** |

Note that, although my targeted training data focused only on the
`equation_symbol_cipher` task, I did not include replay anchors for the
neighboring `equation_digit_ops` task. *(In hindsight, this was a mistake on my
part and likely contributed to Arm B's very poor performance on that task.)*

Before constructing the final training corpora, I created a deterministic,
task-stratified 80/20 split of the original 9,500 rows using seed 12345:

- 7,601 rows in the training split;
- 1,899 rows in the held-out split.

I generated symbol-cipher traces for eligible rows in both splits so that I
could inspect them, but only traces from the training split were used for
fine-tuning. The replay anchors were also sampled only from correct baseline
responses in the training split. This ensured that neither Arm A nor Arm B
contained any held-out rows.

Both arms started from the same Kien rank-32 adapter and used completion-only
supervised fine-tuning:

| Setting | Value |
| --- | --- |
| Base model | `NVIDIA-Nemotron-3-Nano-30B` |
| Warm start | Kien Tinker adapter |
| LoRA rank | 32 |
| Learning rate | `2e-5`, cosine schedule |
| Epochs | 2 |
| Effective batch size | 16 |
| Precision | BF16 |
| Maximum sequence length | 8192 |
| Seed | 12345 |

I trained on Modal. Evaluation used vLLM and followed the official metric
notebook's prompt construction, deterministic decoding, boxed-answer extraction,
and answer comparison.

## Results

### Full 9,500-row results

| Task | Kien baseline | Arm A | Arm B |
| --- | ---: | ---: | ---: |
| **Overall** | **8,214/9,500 (86.46%)** | **8,081/9,500 (85.06%)** | **7,608/9,500 (80.08%)** |
| bit_manipulation | 1,261/1,602 (78.71%) | 1,224/1,602 (76.40%) | 1,233/1,602 (76.97%) |
| cipher_text | 1,565/1,576 (99.30%) | 1,557/1,576 (98.79%) | 1,553/1,576 (98.54%) |
| equation_digit_ops | 563/732 (76.91%) | 534/732 (72.95%) | 63/732 (8.61%) |
| equation_symbol_cipher | 66/823 (8.02%) | 10/823 (1.22%) | 1/823 (0.12%) |
| gravity | 1,593/1,597 (99.75%) | 1,593/1,597 (99.75%) | 1,591/1,597 (99.62%) |
| roman_numeral | 1,576/1,576 (100.00%) | 1,576/1,576 (100.00%) | 1,576/1,576 (100.00%) |
| unit_conversion | 1,590/1,594 (99.75%) | 1,587/1,594 (99.56%) | 1,591/1,594 (99.81%) |

### Held-out 1,899-row split

| Task | Kien baseline | Arm A | Arm B |
| --- | ---: | ---: | ---: |
| **Overall** | **1,640/1,899 (86.36%)** | **1,618/1,899 (85.20%)** | **1,514/1,899 (79.73%)** |
| bit_manipulation | 254/320 (79.38%) | 246/320 (76.88%) | 247/320 (77.19%) |
| cipher_text | 314/315 (99.68%) | 312/315 (99.05%) | 310/315 (98.41%) |
| equation_digit_ops | 110/146 (75.34%) | 108/146 (73.97%) | 7/146 (4.79%) |
| equation_symbol_cipher | 12/165 (7.27%) | 2/165 (1.21%) | 0/165 (0.00%) |
| gravity | 318/319 (99.69%) | 318/319 (99.69%) | 317/319 (99.37%) |
| roman_numeral | 315/315 (100.00%) | 315/315 (100.00%) | 315/315 (100.00%) |
| unit_conversion | 317/319 (99.37%) | 317/319 (99.37%) | 318/319 (99.69%) |

The four easiest tasks stayed almost unchanged. Bit manipulation lost a few
points in both arms. The largest differences appeared in the two equation
variants:

- neither arm improved `equation_symbol_cipher`;
- Arm A reduced `equation_digit_ops` moderately;
- Arm B nearly erased `equation_digit_ops`;
- **→ Arm A was consistently much less destructive than Arm B.**

After training, Arm A performed worse than Kien's baseline in my local
evaluation. It also scored one point lower when I submitted both models to
Kaggle: Arm A scored `0.84` publicly, while Kien's baseline scored `0.85`.

However, there was an interesting twist on the private leaderboard. Arm A
scored `0.85` privately, while Kien's baseline scored `0.84`. I finished 309th
and received a bronze medal.

I do not interpret the one-point private/public swap as proof that Arm A is a
better model. The competition evaluation showed some variation, and the local
held-out comparison still favors the Kien baseline. It is nevertheless the
submission that produced my best private score.

## Discussion

### Why did the traces fail?

The task requires the model to discover both an operator and a digit mapping.
My traces included some operator reasoning, but they skipped the hardest part:
searching for a digit mapping that satisfies every equation at the same time.

During training, the mapping was always correct because the Python solver
inserted it. During evaluation, the solver was no longer present. The model had
to generate a mapping by itself.

It often learned to imitate the trace format:

```text
state a digit mapping
perform arithmetic
say that the examples match
produce a boxed answer
```

But it did not learn the missing search:

```text
generate possible mappings
test them against every equation
reject contradictions
keep the globally consistent mapping
```

Sometimes the generated mapping did not satisfy even one example. The model
still continued with arithmetic and wrote that the result was "matching." In
other words, it learned what a completed verification trace looked like without
reliably performing the verification.

For example, on one row Arm A generated:

```text
The consistent digit assignment is
A=1, B=7, C=4, D=3, F=9, G=7, H=3, I=8, J=1.

Check E1: 71 x 47 = 3337; encoded as DHC, matching DHC.
...
Query: 44 - 73 = -29; encoded as IJH, matching IJH.
\boxed{&'\}
```

This mapping is already invalid: `A` and `J` both map to 1, `B` and `G` both
map to 7, and `D` and `H` both map to 3. The claimed encoded checks also do not
follow from the mapping, but the model still writes "matching" and continues to
the boxed answer. The correct answer for this row was `\boxed{@&}`.

The shortcut therefore asked too much. I was hoping the model would infer an
unwritten algorithm from many completed examples. Instead, it learned to assert
the hidden result.

### Why was Arm A less destructive?

For consistency, I will refer to the two tasks as:

- **symbol-cipher task**: `equation_symbol_cipher`
- **digit-operations task**: `equation_digit_ops`

Although their prompts look similar, Kien's baseline originally uses very
different reasoning procedures for them.

#### Kien's baseline

For the **symbol-cipher task**, Kien usually begins by replacing symbols with
letters:

```text
We need to infer the transformation rule from the examples.

First, let me assign letters to each symbol:
` -> A
! -> B
[ -> C
...

Converting examples to letter form:
...

Operator x: unknown
Operator y: unknown
```

It often performs the setup correctly but fails to discover the hidden
operators and digit mapping. It then defaults to concatenation and produces the
wrong answer.

For the **digit-operations task**, Kien uses a much longer operation search:

```text
We need to infer the transformation rule from the examples.

Looking at the input of the examples...
Looking at the operators...
Looking at operator 「!」...

Trying common operations:
  concatenation: wrong
  reverse concatenation: wrong
  addition: wrong
  absolute difference: 12 match, correct
  subtraction: 12 match, but skipping
  ...

Applying to 21!34:
  absolute difference f(21, 34) = 13

\boxed{13}
```

Here, the search is the reasoning. Kien tries different operations, calculates
their results, rejects the incorrect ones, and applies the matching operation to
the question.

#### Arm A

For the **symbol-cipher task**, Arm A retains Kien's original beginning:

```text
We need to infer the transformation rule from the examples.

First, let me assign letters to each symbol:
...

Converting examples to letter form:
...
```

It then switches to my corrected continuation:

```text
Operator x: multiplication, then add one
Operator y: reversed subtraction

The consistent digit assignment is:
A=5, B=6, C=4, ...

Check E1: ...
Check E2: ...
```

For the **digit-operations task**, Arm A continues to use Kien's original
reasoning procedure:

```text
We need to infer the transformation rule from the examples.

Looking at the operators...
Trying common operations...
Trying rare operations...
Applying the matching operation to the question...
```

All 732 Arm A completions for the digit-operations task began with Kien's
original opener. Arm A therefore preserved the model's existing
operation-search behavior even though it was trained only on the symbol-cipher
task.

#### Arm B

For the **symbol-cipher task**, Arm B was trained to use a completely new
structure:

```text
I will solve the symbol equation directly as a digit cipher.

Label the examples E1-E4.
Anchor: E1 is the most constraining equation.
Remaining candidates: ...
Operator * is multiplication.
Digit assignment: ...
```

For the **digit-operations task**, Arm B unexpectedly began using almost the
same structure:

```text
I will solve the symbol equation directly as a digit-wise operation.

Label the examples as Eq1, Eq2, Eq3.
Anchor: Eq1 is the most constraining usable equation
with 42 surviving candidates.

Remaining Eq1 candidates: 42.
Best match: Eq1: 42.
The operator is digit multiplication.

Digit multiplication on 21!34:
2*3=6, 1*4=4, result 64.

\boxed{64}
```

The correct answer was `13`.

I expected Arm B to retain Kien's original operation search and perhaps express
it using the new wording. Instead, its completions for the digit-operations task
closely followed the procedure and wording from my symbol-cipher training
traces. That procedure did not fit the digit-operations task. Arm B mentioned
anchors, candidates, and a best match, but it no longer performed Kien's long
operation search. None of its 732 digit-operations completions began with Kien's
original opener.

Therefore, although both arms were trained only on the symbol-cipher task:

- Arm A continued to produce Kien-style operation-search traces for the
  digit-operations task.
- Arm B instead produced traces that closely resembled the new symbol-cipher
  training format, which performed very poorly on the digit-operations task.

### My Interpretations

This is my interpretation of the experiment, which I admit is somewhat
speculative. Originally, I imagined prefix surgery as a way to target exactly
where the model begins making a mistake. If the beginning of its response is
already correct, there is no need to train it to generate that part again.
Instead, I keep the correct prefix as context and apply the completion loss only
to the corrected continuation. My hope was that targeting the model's error
directly would make fine-tuning more efficient, since the training signal would
focus on the behavior that actually needs to change. However, after completing
the experiment, I started thinking about this idea more broadly. In particular,
I wondered whether the format in which a model expresses its reasoning and its
actual reasoning capability are separate, or whether they are intertwined.

There is some evidence that particular model behaviors can be changed without
completely changing the model's other capabilities. For example,
[Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) shows that
a targeted behavior can be steered through activation changes while largely
preserving other capabilities. This does not directly show that reasoning and
wording are independent, but it motivated the way I initially thought about the
problem. My expectation for Arm B was that changing the wording and structure
of the traces might change how the model expressed its reasoning without
destroying the reasoning procedure it already knew. In other words, I expected
it to retain Kien's existing reasoning for the digit-operations task, perhaps
expressed using the new trace format. However, that is not what I observed. Arm
A continued to produce Kien-style operation-search traces for the
digit-operations task. Arm B instead began producing traces that resembled my
new symbol-cipher training format, even when solving the very different
digit-operations task. Its performance on that task then collapsed. This does
not prove that reasoning capability and wording are fundamentally inseparable.
Arm A and Arm B differed in more than wording, and this experiment did not
examine the model's internal representations. A more cautious interpretation is
that, in this setting, the model learned a strong association between a
particular reasoning procedure and the format in which that procedure was
expressed. It is also possible that the training setup or the model's capacity
made it difficult to preserve one while changing the other.

Still, this suggests an interesting direction for future work: developing
fine-tuning methods that can correct a specific reasoning decision without
changing the model's existing reasoning format and surrounding behavior. If
reasoning capability could be targeted independently of its wording and format,
fine-tuning might become more efficient by focusing directly on what needs to
change while preserving what the model already knows. Such a method might
require less training data, less training effort, and fewer updates to behavior
that is already correct. This idea remains speculative, but it is the question
I found most interesting after completing the experiment: can a model's
reasoning be corrected independently of the language and structure through
which that reasoning is expressed?

## Sources and Special Thanks

- **[@afr1ste](https://www.kaggle.com/afr1ste) and [@kienngx](https://www.kaggle.com/kienngx)**:
  [adapter guide](https://www.kaggle.com/code/afr1ste/nemotron-0-86-tinker-adapter-guide),
  [public adapter](https://www.kaggle.com/models/kienngx/nemotron-nano-30b-trained),
  and [submission notebook](https://www.kaggle.com/code/kienngx/nvidia-nemotron-trained-models-submission).
  These provided the baseline and warm start used throughout the project.
- **[@lkevincc](https://www.kaggle.com/lkevincc)**:
  [solver discussion](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/698293)
  and [solver repository](https://github.com/lkevincc0/kaggle-nemotron-equation-symbolic).
  This was the foundation of my symbol-cipher solver.
- **[@kemshim](https://www.kaggle.com/kemshim)**:
  [equation-symbol discussion](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/703479).
  This discussion noted how cipher-heavy training can cause forgetting in other
  task categories.
- **[@huikang](https://www.kaggle.com/huikang)**:
  [open progress write-up](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915)
  and [public repository](https://github.com/tonghuikang/nemotron).
  Tong Hui Kang's work formed the foundation of many community approaches to
  deterministic reasoning traces and adapter training.
- **Kaggle's [official metric notebook](https://www.kaggle.com/code/metric/nvidia-nemotron-metric)**:
  prompt construction, decoding, answer extraction, and evaluation behavior.
