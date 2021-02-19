---
layout: post
title: "Day20. Self-supervised Pre-training Models"
subtitle: "GPT-n, BERT, ALBERT, ELECTRA"
date: 2021-02-19 23:55:12+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> Transformer의 self-attention 구조는 NLP분야를 비롯한 다양한 분야에 지대한 영향을 주었다. 높은 성능을 보이는 많은 모델들이 self-attention구조를 채택하게 되었는데,
  오늘은 self-attention구조에 기반한 다양한 pre-training model에 대해 알아보도록 한다.  
  
아래와 같은 순서로 작성하였다.  
- [Recent Trends](#recent-trends)
- [GPT-1/BERT](#gpt-1bert)
    - [GPT-1](#gpt-1)
    - [BERT](#bert)
    - [GPT-1 vs BERT](#gpt-1-vs-bert)
    - [Machine Reading Comprehension (MRC), Question Answering](#machine-reading-comprehension-mrc-question-answering)
    - [BERT: SQuAD 1.1/2.0](#bert-squad-1120)
- [작성중](#작성중)
    - [GPT-2 / GPT-3](#gpt-2--gpt-3)
    - [ALBERT](#albert)
    - [ELECTRA](#electra)
    - [모델 경량화](#모델-경량화)
    - [Graph theory와 NLP의 융합](#graph-theory와-nlp의-융합)
- [Reference](#reference)


<br/>

## Recent Trends
현재 많은 모델들이 transformer의 self-attention 구조를 **깊게 쌓아**, 자기 지도 학습의 프레임워크로 활용되고 있다. 
추후 설명하겠지만, 여기서 프레임워크는 **pre-training된 구조를 말하며 이를 별도의 finetuning 없이도 원하는 task에 활용할 수 있기 때문에 자기 지도 학습으로 이해할 수 있다.**  
  
한편 자연어 처리뿐만이 아니라 여타 많은 분야에서도 이 self-supervise, self-attention 구조를 활용하고 있다. 
  
자연어 처리에서 아직까지의 한계점은 greedy decoding을 근본적으로 해결하지 못한다는 것이다. 
단어 생성시에는 왼쪽부터 하나하나 생성해야하며 sequence 하나를 한 번에 생성하는 방법에 대해서는 아직까지 연구가 진행중이다.  
  
<br />

## GPT-1/BERT
NLP에서의 self-supervised pre-training 모델의 원조격으로 GPT-1/BERT를 들 수 있다.

<br />

#### GPT-1
GPT-1(Generative Pre-training)에서는 \<S\>, \<E\>, \$ 등의 다양한 special token을 활용하여 fine-tuning시의 성능을 극대화한다.   
  
또한 pre-training 구조를 활용하였는데, 이 구조를 활용하면 **미리 학습된 일부 모델에 fine-tuning을 위한 layer만 덧붙여 하나의 모델을 다양한 task에 활용할 수 있다는 장점이 있다.**  
  
![GPT-1](/img/posts/20-1.png){: width="100%" height="100%"}{: .center}   
먼저 앞서 언급한 special token에 대해 알아보면, 기존처럼 문장의 시작에는 Start token을 넣어주고 위 그림에서는 문장의 끝에 **Extract 토큰**을 넣어주었다. 
여기서 이 Extract token은 EoS의 기능 뿐만 아니라 **우리가 원하는 downward task의 query벡터로 활용**된다.  
  
에를 들어 사진의 첫번째 task인 classification 문제를 푼다고 하면, transformer 구조의 마지막 output으로 나온 extract token을 별도의 linear layer에 통과시켜 분류를 수행한다.  
  
두번째 task인 entailment에는 Delim(delimiter) token이 활용되는데, 이것은 서로 다른 두 문장을 이어주는 역할을 한다. 
두 문장을 각각 넣지 않고 Delim 토큰을 활용해 한꺼번에 넣어 두 문장의 논리적 관계(참/거짓)을 파악한다. 이것 역시도 마지막 Extract token을 finetuning된 linear layer에 통과시켜 정답을 얻을 수 있다.  
  
이러한 구조의 장점은, **같은 transformer 구조를 별도의 학습 없이 여러 task에서 활용할 수 있다는 것이다.**
우리는 downward task를 위한 마지막 linear layer만 별도로 학습시켜서 우리가 원하는 task에 활용하면 된다. 
여기서 transformer 구조 부분이 미리 학습되어 활용할 수 있다는 의미로 pre-training model, 그 뒤 linear layer 부분은 finetuning model이라고 부른다.  
  
다만 구조를 더 깊게 들여다보면 사실 transformer(pre-training model) 부분도 아예 학습을 안하지는 않는다.
다만 finetuning 부분에 비해 상대적으로 learning rate를 매우 작게 주어 거의 학습을 시키지 않고, finetuning 부분에 learning rate를 크게 주어 이 부분을 중점적으로 학습시킨다.  
  
이 때 **수행하고자하는 task에 대한 데이터가 거의 없을 때 pre-training model만 대규모의 데이터로 학습시킬 수 있다면 어느정도 target task에도 보장되는 성능이 있다.**
즉, pre-training model의 지식을 finetuning 부분에 자연스럽게 전이학습시킬 수 있다.  
  
활용한 구조를 더 자세히 보면 GPT-1에서는 12개의 decoder-only transformer layer를 활용하였고, multihead의 갯수는 12개, 인코딩 벡터의 차원은 768차원으로 해주었다. 
또한 ReLU와 비슷한 생김새를 가진 GELU라는 activation unit을 활용하였다.
pre-training 단계에서는 language modeling 즉 이전과 같은 text prediction(seq2seq에서처럼)으로 transformer 모델을 학습시킨다. 

<br />

#### BERT 
![BERT](/img/posts/20-2.png){: width="100%" height="100%"}{: .center}   
BERT(Bidirectional Encoder Representations from Transformers)에서는 모델이 학습할 때 이전과 같이 next word language modeling(다음 단어 예측)이 아니라, **일부 단어를 가려놓고(마스킹) 이를 맞추는 방식의 language modeling을 활용한다.**  
  
GPT-1에서는 **masked** multihead attention을 활용하기 때문에 **앞쪽 단어만을 보고 뒷단어를 예측해야한다는 한계점이 존재했다.**
어떤 단어를 예측할 때 뒤에 오는 단어들의 문맥을 고려하지 못하고 학습하기 때문에 실제 다른 downward task를 수행할 때도 성능이 떨어질 우려가 있다.
그렇다고 이전에 썼던 biLSTM을 쓰면, cheating의 우려가 생긴다.
  
BERT에서는 따라서, 학습을 할 때 전체 글에서 일정 퍼센트만큼의 **단어를 가려놓고 그 단어가 무엇인지 맞히는 학습(Masked Language Model, MLM)**을 하게 된다. 
여기서 몇 퍼센트를 가릴지도 하나의 hyperparameter가 되는데, 일반적으로 15%가 최적이라고 알려져있다. 
한편 여기서 15%를 다 가리게 되면 실제 main task 수행시 inference에서 들어오는 문장과 괴리가 있을 수 있으므로(실제 문장에는 mask가 없다), 15%의 candidate도 아래와 같이 역할을 나눈다.
1. 80%는 실제로 masking을 한다.
2. 10%는 임의의 단어(random word)로 바꾼다.
3. 10%는 바꾸지 않고 그대로 둔다.  
  
2번 항목에 해당하는 단어에 대하여, 모델이 어떤 단어가 바뀐(잘못된) 단어가 아니라는 소신 역시 가지게 해야 하므로 3번과 같이 바꾸지 않고 그대로 두는 단어도 둔다. 
근데 사실 이 부분엔 여전히 어폐가 느껴진다. 15%를 가릴건데 그 중 10%는 또 안가린다는게 무슨 말인지.. :cry: 일단은 그냥 원논문에서는 그렇게 구현했다는 점만 기억하자.   
  
또한 language modeling 뿐만아니라, 두 문장 A, B를 주고 B가 A의 뒤에 올 수 있는지 **분류하는 학습(Next Sentence Prediction, NSP)**도 하게 된다.
이 task는 binary classification task가 될 것이며, 모델은 Label로 IsNext 혹은 NotNext를 내놓게 될 것이다.
그리고 GPT-1에서의 special token을 BERT에서도 비슷하게 활용하는데, 앞서 본 NSP에서의 classification task를 위해 
이번에는 문장 맨 앞에(GPT-1에서는 맨 뒤에 Extract token을 놓았다) CLS(classification) token을 두어 이 token에 대한 output을 분류에 활용한다.
CLS token의 output은 layer에 통과되어 분류를 위한 결과를 내놓게 된다.  
   
![BERT_segment_embedding](/img/posts/20-3.png){: width="100%" height="100%"}{: .center}   
위 두 학습에 더불어 positional encoding시 **SEP(seperate) 토큰으로 나눠진 문장이 있으면 각 문장에 별도의 위치정보를 주입해주기 위해 segment embedding을 추가적으로 더해주었다.**
**그리고 BERT에서는 positional encoding 자체도 기존 주기함수가 아니라 별도의 학습을 통해 구하여 더해주었다.**
  
BERT에서는 base model에 self-attention layer 12개, multihead 12개, 인코딩 벡터의 차원을 768개로 두었으며(GPT-1과 동일)
보다 성능을 끌어올린 large model에서는 self-attention layer 24개, multihead 16개, 인코딩 벡터의 차원을 1024개로 주었다.
그리고 데이터로써 byte pair encoding의 변형 알고리즘인 WordPiece model을 활용하여 인코딩된 WordPiece embedding 30000개를 활용하였다.

<br />

#### GPT-1 vs BERT
BERT 모델은 상대적으로 GPT-1에 비해 더 나은 성능을 보여주었다.
어떻게 보면 당연할 수도 있는게, GPT가 제시될 당시 GPT는 8억 개의 word로 학습되었고 BERT는 25억개의 word로 학습되었다.

또한 batch size 역시 GPT가 32000 words, BERT가 128000 words로 BERT가 훨씬 컸다. (보통 큰 사이즈의 배치가 학습에 더 좋다)  
  
한편, GPT는 모든 fine-tuning task에서 똑같이 learning rate를 5e-5로 주었으나 BERT에서는 각 task에서 별도의 learning rate를 두고 fine-tuning 단을 학습시켰다.

<br />

#### Machine Reading Comprehension (MRC), Question Answering
모든 downward task가 MRC 기반(독해 기반) 질의응답으로 이루어질 수 있다는 내용의 논문이 발표되었다.  
  
예를 들어, 문서의 topic이 필요하다면 별도의 fine-tuning 없이 'What is the topic?' 이라는 질문에 대한 응답으로 원하는 답을 얻을 수 있다. 
이에 따르면 결국 별도의 fine-tuning 과정이 생략될 수 있다. 다만 이렇게 되면 pre-training이 더 무거워질것 같기는 하다.

<br />

#### BERT: SQuAD 1.1/2.0
실제로 많은 질의응답 데이터 등을 이용해 BERT를 위에서 언급한 것처럼 질의응답 기반 모델로 발전시킬 수 있다. 
이를 위해 SQuAD(Stanford Question Answering Dataset)라는 크라우드 소싱 기반 데이터가 활용될 수 있다.  
  
SQuAD 1.1 데이터 셋을 활용하여 학습되는 BERT에서는 먼저 질문을 던지면 그 질문에 대한 답이 주어진 문장 어딘가에 있다는 가정 하에, BERT 모델은 정답에 해당되는 단어 sequence의 **첫번째 위치와 마지막 위치**를 예측한다. 
모든 단어를 self-attention에 통과시켜 나온 output vector를 최종적으로 linear layer에 통과시켜 scalar 값을 얻고, 이에 softmax를 적용하여 각 위치를 예측한다.  
  
여기서 추가적으로 필요하게 되는 parameter는 이 output vector를 통과시키는 **첫번째 위치 예측을 위한 레이어의 가중치**, 그리고 **마지막 위치 예측을 위한 레이어의 가중치**로
단 2개 layer의 parameter만 추가되면 우리는 이러한 질의응답 예측이 가능하다.  
  
SQuAD 2.2 데이터 셋을 활용하여 학습되는 BERT에서는 질문에 대한 답이 있는지 없는지부터 판단한다(binary classification).
만약 답이 있으면 아까 1.1에서와 같은 task를 또 수행하고, 답이 없으면 No answer에 해당하는 label을 출력한다.
classification에는 앞에서 언급했던것처럼 CLS token을 이용한다.  
  
비슷한 유형으로, 예제 문장을 주고 이 다음에 올 문장을 4지선다로 고르는 문제가 주어져도, 
예제 문장과 이 4개의 문장을 각각 concat하여 BERT를 통해 해결할 수 있다.  
  
concat한 벡터가 BERT를 통과하여 나온 encoding CLS token을 linear layer에 통과시켜 scalar 값을 얻는다.
이걸 각 문장에 대해 수행하면 총 4개의 scalar 값을 얻을 수 있는데, 이를 softmax에 통과시켜 훈련시킬 수 있으며 이 값을 통해 답을 예측할 수 있다.  
  
지금 소개한 pre-training model(GPT-1, BERT)들은 모델 사이즈를 늘리면 늘릴수록 무궁무진하게 계속 개선된다는 특징이 있다.
![BERT_ablation](/img/posts/20-4.png){: width="80%" height="80%"}{: .center}   
물론 위 그래프처럼 후반부로 갈수록 상승폭이 줄어들긴 하지만, 리소스(GPU)만 많다면 모델의 성능을 무궁무진하게 개선할 수 있다는 점을 알 수 있다.  
  
특히 최근에는 GPT 모델이 GPT-3까지 발전하면서 위와 같은 특성을 유지하면서도 성능이 대폭 개선된 모델이 생겨나게 되었는데,
이로 인해 model size 만능론이 등장하면서 리소스가 부족한 많은 연구자들을 슬프게 만들기도 했다.

<br />

## 작성중
#### GPT-2 / GPT-3
#### ALBERT
#### ELECTRA
#### 모델 경량화
#### Graph theory와 NLP의 융합
<br />

## Reference  
[ALBERT 논문 Review](https://y-rok.github.io/nlp/2019/10/23/albert.html)  
[퓨샷 러닝(few-shot learning)](https://www.kakaobrain.com/blog/106)  