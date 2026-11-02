\# Local AI Energy Measurement — Experimental Results



\## 1. Experiment Overview



\*\*Project:\*\* AI Energy Lab — Energy, Cost and Sustainability Analysis of AI Systems  

\*\*Backend:\*\* Ollama  

\*\*Model:\*\* llama3.2:1b  

\*\*Hardware:\*\* AMD Ryzen 5 7520U with Radeon Graphics  

\*\*CPU:\*\* 4 cores, 8 logical processors  

\*\*Operating environment:\*\* Windows  

\*\*Repetitions:\*\* 3 per prompt  

\*\*Target output:\*\* 30 tokens per run



The purpose of this experiment was to compare the estimated energy consumption and inference performance of a small language model under different input-context sizes.



\## 2. Measurement Methodology



CPU power was estimated using a CPU-utilisation-based model.



The model uses the following provisional parameters:



\- Estimated CPU idle power: 3 W

\- CPU TDP assumption: 15 W

\- Sampling interval: 0.25 seconds

\- Idle baseline measurement duration: 10 seconds



The experiment did not use direct CPU package energy measurement or an external power meter. NVIDIA GPU measurements were unavailable on this machine.



\*\*Important limitation:\*\* The reported power and energy values are model-based estimates, not direct electrical measurements. The assumptions may not accurately represent the laptop's actual power consumption.



\## 3. Experimental Results



Each prompt was tested three times, generating 30 output tokens per run.



| Metric | Short | Medium | Long context |

|---|---:|---:|---:|

| Average input tokens | 41.33 | 60.67 | 1,665 |

| Average output tokens | 30 | 30 | 30 |

| Average generation speed (tokens/s) | 19.52 | 18.88 | 14.06 |

| Average estimated power (W) | 8.83 | 8.32 | 11.28 |

| Average estimated energy (J) | 34.57 | 35.15 | 381.96 |

| Average energy per output token (J) | 0.551 | 0.527 | 7.531 |



\## 4. Interpretation



The short and medium prompts showed similar estimated energy consumption, despite their difference in input size.



The long-context prompt contained approximately 1,665 input tokens, compared with approximately 41 tokens for the short prompt. It required considerably more processing time and estimated energy.



The average generation speed also decreased from 19.52 tokens per second for the short prompt to 14.06 tokens per second for the long-context prompt.



These findings indicate that input-context size can substantially affect inference runtime and estimated energy consumption.



However, the experiment compares different workload sizes. It does not isolate the effect of context length under otherwise identical computational conditions.



\## 5. Limitations



1\. CPU power was estimated using utilisation rather than measured directly.

2\. The 15 W TDP and 3 W idle values are provisional model assumptions.

3\. GPU power measurements were unavailable.

4\. Only three repetitions were conducted for each prompt.

5\. The results are specific to the tested hardware, model, backend and workload.

6\. The experiment does not establish a universal relationship between context length and energy consumption.



\## 6. Conclusion



The preliminary experiment demonstrates that larger input contexts can substantially increase the estimated energy required for local language-model inference.



The results provide a useful baseline for further investigation. Future work should include more repetitions, validated hardware power measurements, additional model sizes and comparisons with cloud inference.



\## 7. Source Data



The raw benchmark CSV is:



`data/results/ollama\_20261102\_123029.csv`



The experiment used three repetitions per prompt and a target output length of 30 tokens.

