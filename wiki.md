# Wiki

## Related Work


### [Unsupervised Real-Time Hallucination Detection based on the Internal States of Large Language Models](https://arxiv.org/abs/2403.06448) (Su et al., 2024)
**Abstract**: Hallucinations in large language models (LLMs) refer to the phenomenon of LLMs producing responses that are coherent yet factually inaccurate. This issue undermines the effectiveness of LLMs in practical applications, necessitating research into detecting and mitigating hallucinations of LLMs. Previous studies have mainly concentrated on post-processing techniques for hallucination detection, which tend to be computationally intensive and limited in effectiveness due to their separation from the LLM's inference process. To overcome these limitations, we introduce MIND, an unsupervised training framework that leverages the internal states of LLMs for real-time hallucination detection without requiring manual annotations. Additionally, we present HELM, a new benchmark for evaluating hallucination detection across multiple LLMs, featuring diverse LLM outputs and the internal states of LLMs during their inference process. Our experiments demonstrate that MIND outperforms existing state-of-the-art methods in hallucination detection.


### [Enhancing Uncertainty-Based Hallucination Detection with Stronger Focus](https://arxiv.org/abs/2311.13230) (Zhang et al., 2023)
**Abstract**: Large Language Models (LLMs) have gained significant popularity for their impressive per- formance across diverse fields. However, LLMs are prone to hallucinate untruthful or nonsensical outputs that fail to meet user expec- tations in many real-world applications. Exist- ing works for detecting hallucinations in LLMs either rely on external knowledge for reference retrieval or require sampling multiple responses from the LLM for consistency verification, making these methods costly and inefficient. In this paper, we propose a novel reference- free, uncertainty-based method for detecting hallucinations in LLMs. Our approach imitates human focus in factuality checking from three aspects: 1) focus on the most informative and important keywords in the given text; 2) focus on the unreliable tokens in historical context which may lead to a cascade of hallucinations; and 3) focus on the token properties such as token type and token frequency. Experimen- tal results on relevant datasets demonstrate the effectiveness of our proposed method, which achieves state-of-the-art performance across all the evaluation metrics and eliminates the need for additional information.


### [HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models](https://arxiv.org/abs/2305.11747) (Li et al., 2023d)
**Abstract**: Large language models (LLMs), such as ChatGPT, are prone to generate hallucinations, i.e., content that conflicts with the source or cannot be verified by the factual knowledge. To understand what types of content and to which extent LLMs are apt to hallucinate, we introduce the Hallucination Evaluation benchmark for Large Language Models (HaluEval), a large collection of generated and human-annotated hallucinated samples for evaluating the performance of LLMs in recognizing hallucination. To generate these samples, we propose a ChatGPT-based two-step framework, i.e., sampling-then-filtering. Besides, we also hire some human labelers to annotate the hallucinations in ChatGPT responses. The empirical results suggest that ChatGPT is likely to generate hallucinated content in specific topics by fabricating unverifiable information (i.e., about 19.5% responses). Moreover, existing LLMs face great challenges in recognizing the hallucinations in texts. However, our experiments also prove that providing external knowledge or adding reasoning steps can help LLMs recognize hallucinations. Our benchmark can be accessed at [this https URL](https://github.com/RUCAIBox/HaluEval).


### [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896) (Manakul et al., 2023)
**Abstract**: Generative Large Language Models (LLMs) such as GPT-3 are capable of generating highly fluent responses to a wide variety of user prompts. However, LLMs are known to hallucinate facts and make non-factual statements which can undermine trust in their output. Existing fact-checking approaches either require access to the output probability distribution (which may not be available for systems such as ChatGPT) or external databases that are interfaced via separate, often complex, modules. In this work, we propose "SelfCheckGPT", a simple sampling-based approach that can be used to fact-check the responses of black-box models in a zero-resource fashion, i.e. without an external database. SelfCheckGPT leverages the simple idea that if an LLM has knowledge of a given concept, sampled responses are likely to be similar and contain consistent facts. However, for hallucinated facts, stochastically sampled responses are likely to diverge and contradict one another. We investigate this approach by using GPT-3 to generate passages about individuals from the WikiBio dataset, and manually annotate the factuality of the generated passages. We demonstrate that SelfCheckGPT can: i) detect non-factual and factual sentences; and ii) rank passages in terms of factuality. We compare our approach to several baselines and show that our approach has considerably higher AUC-PR scores in sentence-level hallucination detection and higher correlation scores in passage-level factuality assessment compared to grey-box methods.


### [The Internal State of an LLM Knows When It's Lying](https://arxiv.org/abs/2304.13734) (Azaria and Mitchel, 2023)
**Abstract**: While Large Language Models (LLMs) have shown exceptional performance in various tasks, one of their most prominent drawbacks is generating inaccurate or false information with a confident tone. In this paper, we provide evidence that the LLM's internal state can be used to reveal the truthfulness of statements. This includes both statements provided to the LLM, and statements that the LLM itself generates. Our approach is to train a classifier that outputs the probability that a statement is truthful, based on the hidden layer activations of the LLM as it reads or generates the statement. Experiments demonstrate that given a set of test sentences, of which half are true and half false, our trained classifier achieves an average of 71\% to 83\% accuracy labeling which sentences are true versus false, depending on the LLM base model. Furthermore, we explore the relationship between our classifier's performance and approaches based on the probability assigned to the sentence by the LLM. We show that while LLM-assigned sentence probability is related to sentence truthfulness, this probability is also dependent on sentence length and the frequencies of words in the sentence, resulting in our trained classifier providing a more reliable approach to detecting truthfulness, highlighting its potential to enhance the reliability of LLM-generated content and its practical applicability in real-world scenarios.


### [Do Large Language Models Know What They Don't Know?](https://arxiv.org/abs/2305.18153) (Yin et al., 2023)
**Abstract**: Large language models (LLMs) have a wealth of knowledge that allows them to excel in various Natural Language Processing (NLP) tasks. Current research focuses on enhancing their performance within their existing knowledge. Despite their vast knowledge, LLMs are still limited by the amount of information they can accommodate and comprehend. Therefore, the ability to understand their own limitations on the unknows, referred to as self-knowledge, is of paramount importance. This study aims to evaluate LLMs' self-knowledge by assessing their ability to identify unanswerable or unknowable questions. We introduce an automated methodology to detect uncertainty in the responses of these models, providing a novel measure of their self-knowledge. We further introduce a unique dataset, SelfAware, consisting of unanswerable questions from five diverse categories and their answerable counterparts. Our extensive analysis, involving 20 LLMs including GPT-3, InstructGPT, and LLaMA, discovering an intrinsic capacity for self-knowledge within these models. Moreover, we demonstrate that in-context learning and instruction tuning can further enhance this self-knowledge. Despite this promising insight, our findings also highlight a considerable gap between the capabilities of these models and human proficiency in recognizing the limits of their knowledge.


### [Do Language Models Know When They're Hallucinating References?](https://arxiv.org/abs/2305.18248) (Agrawal et al., 2023)
**Abstract**: State-of-the-art language models (LMs) are notoriously susceptible to generating hallucinated information. Such inaccurate outputs not only undermine the reliability of these models but also limit their use and raise serious concerns about misinformation and propaganda. In this work, we focus on hallucinated book and article references and present them as the "model organism" of language model hallucination research, due to their frequent and easy-to-discern nature. We posit that if a language model cites a particular reference in its output, then it should ideally possess sufficient information about its authors and content, among other relevant details. Using this basic insight, we illustrate that one can identify hallucinated references without ever consulting any external resources, by asking a set of direct or indirect queries to the language model about the references. These queries can be considered as "consistency checks." Our findings highlight that while LMs, including GPT-4, often produce inconsistent author lists for hallucinated references, they also often accurately recall the authors of real references. In this sense, the LM can be said to "know" when it is hallucinating references. Furthermore, these findings show how hallucinated references can be dissected to shed light on their nature. Replication code and results can be found at [this https URL](https://github.com/microsoft/hallucinated-references).


### [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://arxiv.org/abs/2401.00396) (Wu et al., 2023)
**Abstract**: Retrieval-augmented generation (RAG) has become a main technique for alleviating hallucinations in large language models (LLMs). Despite the integration of RAG, LLMs may still present unsupported or contradictory claims to the retrieved contents. In order to develop effective hallucination prevention strategies under RAG, it is important to create benchmark datasets that can measure the extent of hallucination. This paper presents RAGTruth, a corpus tailored for analyzing word-level hallucinations in various domains and tasks within the standard RAG frameworks for LLM applications. RAGTruth comprises nearly 18,000 naturally generated responses from diverse LLMs using RAG. These responses have undergone meticulous manual annotations at both the individual cases and word levels, incorporating evaluations of hallucination intensity. We not only benchmark hallucination frequencies across different LLMs, but also critically assess the effectiveness of several existing hallucination detection methodologies. Furthermore, we show that using a high-quality dataset such as RAGTruth, it is possible to finetune a relatively small LLM and achieve a competitive level of performance in hallucination detection when compared to the existing prompt-based approaches using state-of-the-art large language models such as GPT-4.


### [Towards Better Evaluation of Instruction-Following: A Case-Study in Summarization](https://arxiv.org/abs/2310.08394) (Skopek et al., 2023)
**Abstract**: Despite recent advances, evaluating how well large language models (LLMs) follow user instructions remains an open problem. While evaluation methods of language models have seen a rise in prompt-based approaches, limited work on the correctness of these methods has been conducted. In this work, we perform a meta-evaluation of a variety of metrics to quantify how accurately they measure the instruction-following abilities of LLMs. Our investigation is performed on grounded query-based summarization by collecting a new short-form, real-world dataset riSum, containing 300 document-instruction pairs with 3 answers each. All 900 answers are rated by 3 human annotators. Using riSum, we analyze the agreement between evaluation methods and human judgment. Finally, we propose new LLM-based reference-free evaluation methods that improve upon established baselines and perform on par with costly reference-based metrics that require high-quality summaries.


### [INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection](https://arxiv.org/abs/2402.03744) (Chen et al., 2024)
**Abstract**: Knowledge hallucination have raised widespread concerns for the security and reliability of deployed LLMs. Previous efforts in detecting hallucinations have been employed at logit-level uncertainty estimation or language-level self-consistency evaluation, where the semantic information is inevitably lost during the token-decoding procedure. Thus, we propose to explore the dense semantic information retained within LLMs' \textbf{IN}ternal \textbf{S}tates for halluc\textbf{I}nation \textbf{DE}tection (\textbf{INSIDE}). In particular, a simple yet effective \textbf{EigenScore} metric is proposed to better evaluate responses' self-consistency, which exploits the eigenvalues of responses' covariance matrix to measure the semantic consistency/diversity in the dense embedding space. Furthermore, from the perspective of self-consistent hallucination detection, a test time feature clipping approach is explored to truncate extreme activations in the internal states, which reduces overconfident generations and potentially benefits the detection of overconfident hallucinations. Extensive experiments and ablation studies are performed on several popular LLMs and question-answering (QA) benchmarks, showing the effectiveness of our proposal.


## Data Sets
### Why is it necessary that the data has not been present during training the LLM?   
There is the differenc between answerable and unanswerable questions. The way how an question becomes unanswerable is just be leaving the particular chunk which is needed for answering the question out from the entire context. Then the LLM's output to that question is always regarded as hallucinated, unless it states uncertainty about its knowledge. However, if the LLM generates the right answer it cannot be classified as a hallucination because it probably has been exposed to that knowledge already during training. Apart from that, we want to ensure that every question is answered solely based on the context and without any internal knowledge.  

Note: The dataset do not need to distinguish between answerable and unanswerable questions because this is part of the subsequenty procedure (when the chunk is either given or not given to the LLM).

All these datasets face the problem of containing questions that an LLM might answer solely based on its own "knowledge".


### Why are only new Wikipedia articles used for the new dataset?
Short answer: In order to avoid outdated knowledge.

An example scenario: If we take a sentence change in Wikipedia that previously stated Sultan Kosen as the tallest person in the world. However, this would be altered because Sultan Kosen has passed away. So if I were to ask the LLM without context, it would confidently answer Sultan Kosen. If I wanted to make this question unanswerable by randomly inserting chunks into the prompt, the LLM would probably still confidently answer Sultan Kosen. However, if I were to then cross-check with the new person (using GPT3.5-Turbo), it turns out the answer would be incorrect, making the response hallucinated, even though the LLM answered correctly based on its own knowledge. Thus, internally within the LLM itself, there would be no way to recognize or have indications that uncertainty prevailed and/or the answer was fabricated.

Apart from that, if we consider answerable questions, meaning that the newly updated chunk is part of the context, both facts (context chunk and trained data) would be contradictory which would lead to unknown behaviour of the LLM and a different representation of "hallucinating" in the internal states. It needs to "randomly" choose what knowledge (context or training data) it considers as true and false. That is why we must exclude facts that might change, so the dataset contains only "new" facts that are less likely to contradict with the LLM's knowledge.


### CoQA
CoQA contains 127,000+ questions with answers collected from 8000+ conversations. Each conversation is collected by pairing two crowdworkers to chat about a passage in the form of questions and answers. The unique features of CoQA include 1) the questions are conversational; 2) the answers can be free-form text; 3) each answer also comes with an evidence subsequence highlighted in the passage; and 4) the passages are collected from seven diverse domains. CoQA has a lot of challenging phenomena not present in existing reading comprehension datasets, e.g., coreference and pragmatic reasoning.
- GitHub-Page: https://stanfordnlp.github.io/coqa/
- Size: 127k questions
- Pros:
  - answers chosen that all four annotators agreed on
  - various questions for a single chunk
- Cons:
  - question immediately refer to chunk (high usage of pronouns and short questions; The usage of pronouns might make an LLM not recognize what chunk it needs to pick in order to answer the query. It is also not common in RAG applications.)
  - based on data that might be seen during training


### QuAC
Question Answering in Context is a dataset for modeling, understanding, and participating in information seeking dialog. Data instances consist of an interactive dialog between two crowd workers: (1) a student who poses a sequence of freeform questions to learn as much as possible about a hidden Wikipedia text, and (2) a teacher who answers the questions by providing short excerpts (spans) from the text. QuAC introduces challenges not found in existing machine comprehension datasets: its questions are often more open-ended, unanswerable, or only meaningful within the dialog context.
- HuggingFace: https://huggingface.co/datasets/quac
- Size: 12k rows
- Pros:
  - written by humans
  - answerable and unanswerable
- Cons:
  - high usage of pronouns instead of the subject/person that is talked about
  - almost the same form of questions ("When did they...", "How is he...", ...)
  - based on data that might be seen during training


### HELM
3342 sentences from randomly sampled Wikipedia articles that were continued by LLMs. Humans annotated it either as unverifiable or as verifiable (using Wikipedia and top 20 Google search results). 
- GitHub: https://github.com/oneal2000/MIND/tree/main
- Size: 3342 sentences, 1224 passages
- Pros:
  - sentence level hallucination annotation
  - already contains contextualized embeddings, self-attentions, and hidden-layer activations
- Cons:
  - limited task (continue writing a section of a Wikipedia article correctly)
  - based on data that might be seen during training


### NoMIRACL
Human generated questions with top-k passages retrieved from a RAG pipeline and a value indicating whether each passage is relevant for answering the question.
- HuggingFace: https://huggingface.co/datasets/miracl/nomiracl
- Arxiv: https://arxiv.org/abs/2312.11361
- Size: 56k rows
- Pros:
  - choice of 18 different languages
  - differentation between unanswerable and answerable in terms of RAG
-  Cons:
  - no answers or references to finding the answer in the relevant passage(s)
  - based on data that might be seen during training


### Natural Questions
Questions that might be answerable with the given Wikipedia article or not.
- HuggingFace: https://huggingface.co/datasets/natural_questions
- Examples: https://ai.google.com/research/NaturalQuestions/visualization
- Size: 26,299 rows
- Pros:
  - high quality
  - long and short answers
- Cons:
  - simply asking for facts in the article (no other tasks such as comparing)
  - based on data that might be seen during training
- Extension: CLAP NQ (https://github.com/primeqa/clapnq)


### HotPotQA
Wikipedia-based question-anwers pairs with mutliple contexts needed to answer the question.
- HuggingFace: https://huggingface.co/datasets/hotpot_qa?row=0
- Size: 203k rows
- Pros:
  - no plain asking for a fact (different types: "comparison" and "bridge") -> requires reasoning
  - high quality
  - diverse question formats
- Cons:
  - based on data that might be seen during training


### MMLU
Multiple-choice questions (+ correct answer) covering 57 tasks and 59 topic domains.
- HuggingFace: https://huggingface.co/datasets/cais/mmlu
- Size: 231k rows
- Pros:
  - high quality
  - diverse
- Cons:
  - not based on a context
  - same format
  - based on data that might be seen during training


### SQuAD (v2)
Question and answer pairs based on a certain context from a Wikipedia article
- Huggingface: https://huggingface.co/datasets/rajpurkar/squad_v2
- Size: 142k rows
- Pros:
  - answers are quoted in the context
  - made through crowdworkers
  - answerable and unanswerable
- Cons:
  - simply asking for content in the article (no other tasks such as comparing)
  - always questions with the same structure (no variety)
  - based on data that might be seen during training


### MS MARCO (v. 2.1)
Bing queries paired with a list of answers and passages potentially containing the answer
- Huggingface: https://huggingface.co/datasets/ms_marco
- PapersWithCode: https://paperswithcode.com/paper/ms-marco-a-human-generated-machine-reading
- Size: 1M rows
- Pros:
  - annotated by humans
  - huge variety of topics
- Cons:
  - Question might still be answerable, although it is labeled as "No Answer Present." (shows lack of quality)
  - based on data that might be seen during training
  - questions resemble only the format of bing queries (short, lower case, no punctuation)


### RAGTruth
Word-level annotated hallucinations of outputs generated by llms regarding a certain context and task.
- GitHub: https://github.com/ParticleMedia/RAGTruth
- Size: 18k rows
- Pros:
  - hallucinations labeled word by word
  - annotated by humans
  - different tasks (summarization, question answering, data to text)
- Cons:
  - based on data that might be seen during training
  - "only" LLaMA and Mistral as local LLMs


### SelfAware
This dataset contains 2337 answerable and 1032 unanswerable questions. The unanswerable questions are sourced from knowledge-sharing platforms such as Quora and HowStuffWorks, while the answerable questions come from SQuAD, HotpotQA, and TriviaQA. The dataset is intended for research purposes only. For any commercial use, it is recommended to contact the author for permission.
- GitHub: https://github.com/yinzhangyue/SelfAware
- Size: +3k rows


### HADES
A token-level, reference-free hallucination dataset based on English Wikipedia sentences verified by crowdworkers.
- GitHub: https://github.com/microsoft/HaDes?tab=readme-ov-file