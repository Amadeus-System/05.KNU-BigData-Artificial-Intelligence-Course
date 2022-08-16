# 05.KNU-BigData-Artificial-Intelligence-Course
Repository For My BigData and Artificial Intelligence Course in KNU

https://lelouch0316.tistory.com/entry/%EA%B0%95%EC%9B%90%EB%8C%80%ED%95%99%EA%B5%90-%EC%9C%B5%ED%95%A9%EB%B3%B4%EC%95%88%EC%82%AC%EC%97%85%EB%8B%A8-%EC%A3%BC%EA%B4%80-2020-%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-%EA%B5%90%EC%9C%A1%EA%B3%BC%EC%A0%95-%ED%9B%84%EA%B8%B0?category=1076241

* **Date : 2022-03-28**
* **Last Modified At : 2022-06-29**


# 강원대학교 융합보안사업단 주관 2020 빅데이터 인공지능 교육과정 후기

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/176369686-94918b2b-c914-4ee1-8a14-2f2315bd146f.jpg">
</p>
<center><b></b></center>
<br>

<!-- TOC -->

- [강원대학교 융합보안사업단 주관 2020 빅데이터 인공지능 교육과정 후기](#%EA%B0%95%EC%9B%90%EB%8C%80%ED%95%99%EA%B5%90-%EC%9C%B5%ED%95%A9%EB%B3%B4%EC%95%88%EC%82%AC%EC%97%85%EB%8B%A8-%EC%A3%BC%EA%B4%80-2020-%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-%EA%B5%90%EC%9C%A1%EA%B3%BC%EC%A0%95-%ED%9B%84%EA%B8%B0)
    - [교육과정 동안 배운 것](#%EA%B5%90%EC%9C%A1%EA%B3%BC%EC%A0%95-%EB%8F%99%EC%95%88-%EB%B0%B0%EC%9A%B4-%EA%B2%83)
    - [느낀 점](#%EB%8A%90%EB%82%80-%EC%A0%90)
    - [최종 프로젝트 : "심층 신경망을 활용한 수력댐 강우량 예측"](#%EC%B5%9C%EC%A2%85-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8--%EC%8B%AC%EC%B8%B5-%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%9C-%EC%88%98%EB%A0%A5%EB%8C%90-%EA%B0%95%EC%9A%B0%EB%9F%89-%EC%98%88%EC%B8%A1)
        - [RainNet : Convolutional Neural Network for Radar-based Precipitation Nowcasting](#rainnet--convolutional-neural-network-for-radar-based-precipitation-nowcasting)
        - [RainNet with Skip Connection](#rainnet-with-skip-connection)
        - [Convolution / Pooling](#convolution--pooling)
        - [LSTM Model](#lstm-model)
        - [ConvLSTM Model](#convlstm-model)
        - [Trajectory GRU Model](#trajectory-gru-model)
        - [예측 결과](#%EC%98%88%EC%B8%A1-%EA%B2%B0%EA%B3%BC)
    - [프로젝트 결과](#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B2%B0%EA%B3%BC)

<!-- /TOC -->

이 교육과정은 정말 우연한 기회로 알게 되었다. KNU Everytime 커뮤니티에 어느 날 이런 포스터가 올라왔었던 것이다. 나는 2차례의 연구 경험을 통해 프로그래밍에 대해 어느정도 익숙한 상태였지만, 언제나 독학으로 머신러닝을 공부해왔기 때문에 한번 쯤은 제대로 된 교육과정을 듣고 싶다는 욕심이 있었다. 

학원을 다닐까 생각도 해봤지만, 학원에서 내가 원하는 수준의 교육을 제대로 할 것 같지 않았고, 학원을 다닐 시간이나 비용도 거의 없었다. 그런 나에게 아주 좋은 기회라고 생각했기 때문에 바로 신청하게 되었다.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/160247989-f565b71e-0d83-4689-b5cc-4819a1b188c9.jpg">
</p>
<center><b>빅데이터 인공지능 안내 포스터</b></center>
<br>

당시 커뮤니티에서 해당 교육과정의 홍보담당자가 여러 학생들의 질문에 답변해주면서 상당히 인기를 끌었던 기억이 있다. 경쟁률이 어느 정도 있을 것으로 생각되었으나, 다행히 1차에 한번에 합격하게 되었다!

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/160247986-fb2bb6a8-991a-4410-8366-87652d668571.jpg">
</p>
<center><b>기술인재 1기 합격</b></center>
<br>

나와 같이 신청했던 학부연구생은 불행히도 탈락해서, 나중에 2차 교육과정에 재도전하여 한 달 늦게 수강하게 되었다. 이 때는 훈련수당을 받으며 내가 원하는 인공지능 분야를 (비록 기초수준이지만) 배울 수 있다는 사실에 행복하기만 했다. ㅎㅎ



## 교육과정 동안 배운 것

* Fundamental Python Programming
* Python Library for Data Processing
* Database (SQL)
* Web Programming
* Machine Learning
* Deep Learning
* Deep Learning Framework (Tensorflow, Keras, PyTorch)
* Natural Language Processing (NLP)
* Audio Voice Processing
* Computer Vision
* Final Projects : Precipitation Nowcasting / Sentimental Analysis of KNU Website

위 목록을 보면 알겠지만 **초심자 입장에서는 저것들 중 하나라도 6개월 동안 제대로 배우면 다행**이다. 미리 예습해가는 것은 필수! 

즉, 굉장히 수업진도가 빠르게 나간다는 것을 알 수 있었다. 다행히 나는 데이터베이스와 웹프로그래밍을 제외하면, 나머지에 대해서는 기초정도는 공부한 적이 있었기 때문에 어느정도 수월하게 따라갈 수 있었다. 그러나 평일 하루에 최소 4~5시간은 투자해야 교육과정을 따라갈 수 있었기 때문에 굉장히 바쁜 6개월을 보낼 수 밖에 없었다. 



## 느낀 점

이 교육과정을 듣기 전까지 나의 프로그래밍 경험은 같았다.

* 약 1년 동안의 열통계물리 수치해석 및 머신러닝 응용 졸업연구
* 6개월 동안의 SCOPUS 논문 연구

과거의 나를 떠올려보면 약간 자신감이 과하지 않았나싶다. 겨우 2번의 연구경험으로 나 정도면 꽤 잘하는거 아닐까? 하고 생각했던 것 같다. (ㅋㅋ 흑역사) 사실 이 정도는 누구나 할 수 있는 것일텐데.. 아마도 더닝크루거 곡선의 정상에 위치하지 않았나싶다.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/160247987-a5c787e3-87fd-4948-9b52-f707a7b43665.png">
</p>
<center><b>Dunning–Kruger Effect</b></center>
<br>

교육과정을 들을 당시의 나는 이 교육과정을 통해 무언가 대단한 것을 배울 수 있다고는 생각하지 않고, 대단히 평가절하하고 있었다. 왜냐하면 해당 교육과정이 대학교 주관으로 이루어지는 국비지원 IT 교육의 일종이라고 이해하고 있었고, 그러한 교육과정에 대한 사람들의 부정적인 후기를 인터넷에서 자주 보아왔기 때문이다.

교육과정이 시작되고 실제로 2달 동안은 별로 대단한 것이 없었다. 내가 이미 지겹게 공부했던 Python 기초문법과 라이브러리의 사용법을 배웠다. 약간 특이한 라이브러리는 재미있었다.

가장 흥미가 없었던 것은 데이터베이스와 웹 프로그래밍이었는데, 나는 도대체 이걸 왜 배워야 하는지 이해할 수가 없어서 거의 중도포기할 위기까지 갔었다. 내가 배우고 싶었던 것은 복잡한 데이터 분석이나 머신러닝이었기 때문에, 그게 아닌 쓸데없어 보이는 다른 것을 배우는 것에 대해 도저히 열의를 낼 수 없었다. 

다행히 교육과정 3달에 접어들자 머신러닝을 배우게 되면서, 드디어 내가 공부하고 싶은 내용들이 주를 이루게 되었다. 이 때부터는 확실히 의욕적으로 공부했고, 폭발적으로 지식을 흡수하게 되었다.

교육과정의 특성상 대단히 방대한 주제를 속성으로 다루었지만, 각 주제별 기초 ~ 중급 수준의 코드를 공부하고 넘어가는 것이 생각보다 만만치 않았다. 특히 중급수준의 코드를 다루는 수업시간에는 코드의 구성이 의외로 수준높다고 느껴져서 어렵지만 만족스러웠다. 또한, 내가 잘 모르지만 널리 사용되는 전형적인 문법스타일도 일부 익힐 수 있었다.

상당히 속도감 있게 수업이 진행되었으므로, 교육과정 동안 모든 것을 완벽히 익혔다고는 할 수 없지만 매일 최소 4시간 이상을 규칙적으로 코딩하면서 어느 정도 개발자적인(?) 마인드로 작업을 한다는 것이 어떤 느낌인지 알 수 있었다. 그 전까지는 약간 수학공부하듯 코드 한 줄 한 줄을 정확히 이해하고 넘어가는 것을 중시했는데, 방대한 양의 코드로 작업하다보면 항상 그럴수는 없고, 상황에 따라서는 빌려쓰는 라이브러리나 함수의 Input, Output 관계만으로 그 기능을 추론하고 고수준에서 사용하는 것도 중요하다는 것을 알았다.

최종적으로 교육과정 다섯달째까지의 딥러닝 공부는 상당히 괜찮았다. 교육과정 마지막 부분에서 컴퓨터비전/자연어처리/음성처리와 같은 딥러닝의 주요 분야를 다시 한번 심도있게 배운 것도 괜찮았다. 물론 관련분야 전공자들이 보기에는 어린아이 장난 수준이겠지만, 당시의 초보였던 나에게는 꽤 어렵게 느껴졌다.

교육과정이 딥러닝까지 끝난 이후, 각 분야에서 전문가들이 초빙되어 실무와 연관된 인공지능 기술의 응용가능성에 대해 2주 동안 강의를 들었다. 사실 이 단계는 또 재미없었다. 나는 이런 기술들의 상업적 응용에는 크게 관심없고, 나의 프로그래밍/머신러닝 실력을 심화시킬 수 있는 높은 수준의 지식/기술을 중점적으로 원했기 때문이다. 이런 면은 응용을 중시하지 않는 이론적인 자연계열 학생답다고 볼 수 있다. ㅎㅎ



## 최종 프로젝트 : "심층 신경망을 활용한 수력댐 강우량 예측"

실무관련 강의까지 모두 끝난 이후에는 드디어 IT 교육과정의 꽃(?)이라고 할 수 있는 최종 프로젝트를 하게 되었다. 컴퓨터공학과를 졸업했다는 직장인 CSM씨와 함께 2인 팀으로 Dacon에서 진행되는 수력댐 기후예측 프로젝트를 진행하였다.

**강우예측(Precipitation Nowcasting)** 문제는 인공위성을 통해 얻어진 여러 지역의 레이더 사진들이 이미지의 Sequence로 주어질 때, 그 다음 시점의 강우 이미지를 예측할 수 있는 모델을 만들 수 있는가에 관한 것이었다. 이 대회를 통해서 처음으로 이미지 Sequence를 다루는 딥러닝 문제 유형을 알게 되었다. 참고로 이 때 알게 된 Precipitation Nowcasting이라는 분야는 나중에 타대학의 AI 연구실에 가서 교수님과 대화를 할 때, 이런저런 이야기를 하다가 내가 이것을 말하자 교수님께서 살짝 놀라셨던 기억이 있다. 놀랍게도 그 연구실에서는 기상청으로부터 연구과제를 받아 이쪽 분야의 연구과제를 수행중이었다. 뭐든 열심히 해두면 나중에 연결된다. ㅎㅎ

나와 팀원은 관련 분야의 논문을 10편 정도 읽으며 기초적인 지식을 습득하였다. 나는 ConvLSTM2D 기반의 모델을, 팀원은 RainNet이라는 CNN 및 Upsampling 기반의 모델을 만들었다.

다만 나의 ConvLSTM2D 모델은 지나치게 많은 학습량을 요구했기 때문에, 실질적으로 대회기간 동안 학습을 모두 끝마치기 어려운 상황이었다. 결과적으로 뜻하지 않게 약간의 버스를 타는 것 같아 팀원 분에게 미안했다. 대신 발표 PPT를 만드는 과정에서는 열심히 했다. ㅎㅎ

또한, 같은 연구실 소속의 학부연구생인 PSH씨와 함께 강원대학교 Everytime 커뮤니티에서 Web Crawling을 이용한 **댓글 오염도 분석** 프로젝트를 추가로 동시에 진행하였다. 웹 크롤링의 초반 부분을 내가 조금 알려준 것을 제외하면 나머지 부분은 PSH씨가 좀 더 열심히 했던 것 같다. 나는 자연어처리에는 크게 관심이 없었고, 해당 프로젝트는 수업시간에 공부했던 감성분석 코드에서 데이터 부분만 바꾸면 되는 프로젝트여서 크롤링과 전처리, 시각화 정도만 테스트하고 PSH씨에게 맡겼다.

사실 나는 PSH씨가 어떤 의미가 있는 댓글 오염도 분석을 하겠다는 것인지 마지막까지 이해하지 못했다. ㅡㅡ; 일단 댓글 오염도 분석으로 긍정/부정 판별을 하겠다는 것은 이해했는데, 왜 그런 프로젝트에 마스코트 캐릭터가 필요한 것인가..?? 나는 전혀 이해할 수 없었다. 아마 단순한 딥러닝 프로젝트가 아니라, 뭔가 어플리케이션을 만드는 그러한 기획을 했던 것 같다. 어쨌든 본인이 나름대로 뿌듯하게 진행하는 것 같아서 PPT에서 디자인 부분은 말을 아끼고 하고싶은 대로 하게 했다. ㅎㅎ

여기서 잠깐 최종 프로젝트에 대해 간단히 설명하고 넘어가고자 한다.



### RainNet : Convolutional Neural Network for Radar-based Precipitation Nowcasting

최종 프로젝트로 나와 팀원은 Dacon에서 주최하는 수력댐 강우예측 대회에 참가하기로 하였다. 해당 대회에서는 레이더로 수집된 특정지역의 구름 이미지를 기반으로 다음 시점의 구름 이미지가 어떻게 될 것인지를 예측하는 문제를 다루었다. 우리는 관련분야의 논문을 살펴보고 적당한 모델을 구현해보기로 했다.

다음 그림은 기후예측 문제해결을 목적으로 U-Net 형태를 갖는 FCNs 기반의 Convolutional Neural Network의 구조를 보여준다.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/160360390-3c8c28e2-a45c-410c-b514-b2338cfa034c.png">
</p>
<center><b>RainNet Architecture</b></center>
<br>

여기서 FCNs(Fully Connected Networks)는 Semantic Segmentation에서 Skip-Connection 구조를 차용한 것이다. U-Net은 Biomedical 분야에서 Image Segmentation을 위한 End-to-End 방식의 FCNs 기반 구조를 갖는 모델이었다. 이러한 모델은 모델의 층이 깊어짐에 따라 Input의 공간 정보를 유지한다고 알려져 있었다.


### RainNet with Skip Connection

그럼 Skip-Connection은 무엇일까?

다음 그림과 같이 Down Sampling의 결과인 Local Information과 Up Sampling의 Semantic Information을 결합한 것을 말한다. 

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/160360419-dff6f4b9-fe1d-42b0-9526-10598e1147b5.png">
</p>
<center><b>이전 상태와 현재 입력을 결합하여 Local Information 활용</b></center>
<br>

여기서 Local Information은 모델에 최초로 들어왔던 Input 이미지의 저차원 Pixel 배열로부터 유의미한 특성맵으로 변환된 합성곱 결과에 대해 Pooling을 실행하여 얻어진 일종의 압축정보이다.

Semantic Information은 순차적인 합성곱, 풀링 연산만으로는 모델이 학습할 수 없는 더 추상적인 정보를 의미하는 것 같았다. 또한, Skip-Connection은 기존의 CNN 모델에서 전형적으로 보이는 Sequential한 구조가 아니라, 중간에 다른 연산회로를 레이어를 점프하는 방식으로 삽입함으로써, 역전파(Back-Propagation) 방식으로 학습할 때 Gradient의 전파를 좀 더 원활하게 해주는 특성이 있었다. 결과적으로 일반적인 CNN 모델보다 층이 깊어지면서 이미지의 위치정보를 잃게 되는 단점을 극복할 수 있었다.



### Convolution / Pooling

이 부분은 딥러닝을 공부하면 초반에 배우는 부분이어서 다들 알 것 같다. CNN 기반의 모델들은 초창기 AlexNet 시절부터 발전해오면서 거의 지속적으로 Convolution, (Max)Pooling, ReLU 함수의 사용이 유지되어 왔다.

CNN은 크게 2단계로 구성되는데, 첫 단계는 이미지의 기하학적 특성을 추출하는 Feature Extraction 과정이다.

다음 그림에서 왼쪽은 이미지가 Input으로 입력되었을 때, 모델 내부에서 이루어지는 특성추출 과정을 보여준다. 이미지는 모델 내부로 들어와 특정한 크기의 Convolution Filter와 Element-wise Product를 하고 그 결과를 모두 더하여, 특성맵(Feature map)을 얻게 된다. 이 부분은 수리물리학 또는 공업수학에서 배우는 합성곱 연산의 Discrete한 방식임을 이공계 학생이라면 쉽게 알 수 있다.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/160360432-f5889c7b-d28e-4925-8eff-69b2138ef3be.png">
</p>
<center><b>Convolution and Pooling</b></center>
<br>

이렇게 얻어진 특성맵은 보통 Convolution 연산 이전의 Raw-Pixel 배열보다 좀 더 유의미한 기하학적 정보를 많이 포함하고 있다. 이러한 정보들은 그대로 사용될 수도 있지만, 보통 Pooling이라는 방식으로 일종의 대표값을 추출하여 사용되는 경우가 많다. 

그리고 이러한 Convolution-Pooling 과정이 반복적으로 이루어지다가 최종적으로 이미지의 고차원적 특성을 충분히 잘 설명할 수 있는 벡터로 표현되어 일반적인 FCL에 들어가고, 주어진 라벨을 추정하도록 학습된다.

결국 CNN은 이미지의 국소적 영역에 존재하는 기하학적 특성을 추출하여 고차원적 의미를 추론하기 위해 사용되는 것이며, 이미지 인식 분야에서 전형적으로 사용되는 모델이라 할 수 있다. 오늘날 CNN의 이미지 인식능력은 인간의 시각인식률(약 96%)를 넘어서는 것으로 알려져 있다.



### LSTM Model

CNN을 통해 이미지 인식을 효과적으로 학습할 수 있다는 것을 알게 되었다. 그럼 문제는 모두 풀린 것일까?

사실 그렇지 않다. 왜냐하면 우리가 다루었던 문제는 Image Sequence를 학습하고 그 다음 시점 $t$의 새로운 Image를 예측해야 하는 문제였기 때문이다.

이는 우리의 모델에 들어오는 Input이 이미지 1장이 아니라 여러 이미지의 나열이고, 각 이미지는 전후 사이의 순서에 의미가 있기 때문에, 순서를 학습하는 기능이 필요함을 의미했다.

다음 그림은 LSTM이라는 RNN(Recurrent Neural Network) 계열의 대표적인 모델을 보여준다.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/160360444-904a0ced-da5c-463f-b72c-e3b34b99a203.png">
</p>
<center><b>LSTM Model</b></center>
<br>

LSTM의 내부연산을 보면 다소 복잡해보이는 내부 회로들이 있고, 각각의 회로는 무언가 복잡한 연산을 하는 것처럼 보인다. 조금 어려워보일 수 있지만, 결국 하나의 LSTM 모듈이 하고자 하는 것은, 입력되는 Sequence의 순서를 학습하여 다음에 무엇이 나올지를 예측하려는 것이다. 그리고 초기 Vanila RNN 모델에 비해, Sequence 정보의 중요성을 판별하여 그 다음 시점의 학습까지 보존할지 아닐지를 조율하는 Cell state라는 벡터가 하나 추가된 것 뿐이다.

약간 이야기를 거슬러 올라가자면, 자연어처리 분야에서는 기계번역(Machine Translation)이라는 특정한 언어를 다른 언어로 번역하는 Task가 존재한다. 예를 들면, 한국어 문장 "나는 학교에 간다." 라는 정보가 Word Vector의 Sequence로 입력되면, 영어 문장 "I go to school."을 예측하도록 학습해야 하는 것이다. 이 경우 모델은 Sequence를 입력받아 Sequence를 출력해야 하므로 이와 관련된 문제상황을 보통 Seq2Seq라 부른다. 그리고 Seq2Seq에서 초기에 등장했던 모델이 Vanila RNN이라 불리는 단순한 형태의 Recurrent 방식 모델이었는데, RNN은 입력되는 문장의 길이가 길어지면 앞 부분의 정보를 기억하지 못하는, 일종의 장기기억 손실 문제가 있었다. 그리고 문제를 해결하기 위해 새로 고안된 모델이 이후에 알려지는 LSTM이나 GRU 등의 모델이었던 것이다. 여기서는 편의상 단순한 설명은 모두 알 것이라 가정하고 넘어간다.

즉, LSTM 셀을 연속적으로 연결하여 신경망을 구성하면 시계열(Time Series) 데이터의 패턴 학습에 특화된 모델을 만들 수 있었다. 이러한 모델들은 오랜 과거에 발생했던 사건의 정보까지 놓치지 않고 패턴학습이 가능하다고 알려져 있었다.



### ConvLSTM Model

앞서 배운 CNN과 LSTM은 사실 딥러닝을 조금만 공부하다보면 금방 익숙해지게 되는 개념이다. 이 프로젝트를 하던 당시에도 이 정도의 기초적인 지식은 알고 있었다.

그러나 Precipitation Nowcasting 관련 논문들을 읽으면서, 이 두 가지를 결합한 새로운 모델이 있다는 것을 알게 되었는데 그것이 바로 ConvLSTM이라는 모델이었다.

다음 그림의 왼쪽을 보자.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/160360448-07afe6fe-0c6b-4cae-9f42-ebf8ebf7904f.png">
</p>
<center><b>ConvLSTM Model</b></center>
<br>

앞서 배웠던 LSTM과 대단히 유사한 구조로 보인다. $C_{t}$로 표현되는 Cell state 벡터가 있고, $h_{t}$로 표현되는 Hidden state 벡터도 있다. 그러나 모듈 내부의 연산을 자세히 살펴보면 다른 점이 있다.

우선, Input으로 들어오는 것이 Vector의 Sequence가 아니라, 이미지의 Sequence이다. 즉, 각각의 Time step $t$에서, 해당 시점에 대응되는 한 장의 이미지가 들어온다. 그리고 그 이미지는 LSTM에서 봤던 평범한 벡터 연산이 아니라, 이미지의 특성을 추출하는 Convolution 연산을 한다!! 그리고 중간중간 분포를 조정하기 위한 BatchNormalization(BN)이 삽입되어 있다.

결국, ConvLSTM 모델은 이미지에 특화된 합성곱(Convolution) 연산을 LSTM 모델 내부로 결합시킨 형태로 볼 수 있다. ConvLSTM을 발표한 논문의 주장에 따르면, 이러한 방식의 연산은 Convolution 연산과 LSTM 연산을 따로 나누어 학습할 때보다 더 효과적으로 이미지의 숨겨진 은닉정보를 학습할 수 있다고 한다. 

이 발상은 상당히 인상깊었다. 왜냐하면 딥러닝을 조금 배운 학생이 이미지 Sequence 학습에 대해 떠올릴 수 있는 가장 평범한 생각은, Convolution을 먼저 해서 특성벡터를 추출하고, 이러한 특성벡터를 모아 Sequence로 학습시키는 것이기 때문이다. **즉, Two-Stage에 걸쳐서 서로 다른 종류의 연산을 두 번 하는 것 대신, 두 연산을 결합하여 One-Stage로 학습이 되도록 고안된 것이다!!** 당시에는 몰랐지만, 이는 마치 객체인식 분야에서 Classification과 Localization을 동시에 수행하는 YOLO 같은 One-Stage Detector 모델을 떠올리게 한다.

최종적으로, 시계열 이미지 데이터의 시간적 속성과 공간적 속성을 동시에 학습하는 모델이 만들어질 수 있었다. 위 그림의 오른쪽을 보면 그러한 모델의 도식이 표현되어 있다. 모델이 Input의 정보를 충분히 학습하여 데이터를 잘 설명할 수 있는 특성벡터를 Encoding할 수 있게 되면, Decoding 하는 모델 뒷 부분에서 추출된 특성을 기반으로 데이터를 복원할 수 있었다.



### Trajectory GRU Model

그리고 사실 위의 두 모델 이외에도, 세번째로 준비하던 모델이 있었는데 바로 Trajectory GRU라는 모델이었다.

여기서 GRU는 Gated Recurrent Unit의 약자로, LSTM과 유사하게 Sequence를 학습할 수 있는 RNN 계열의 모델이다. 그리고 이러한 GRU에 대해 ConvLSTM과 유사하게 Convolution 연산을 결합하여 학습하는 모델에 관한 논문이 있었다. 다만, 이 부분은 논문을 보며 이론적인 내용을 조금 공부하였을 뿐 실제로 구현하여 학습하지는 않았기 때문에 지금은 설명을 잘 할 수 없을 것 같다. (나중에 다시 공부해야겠다.)

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/160360455-41f41710-aca5-4cf8-b10a-f157d51f3546.png">
</p>
<center><b>Trajectory GRU Model</b></center>
<br>

Trajectory GRU를 제시한 논문의 주장은 대략 이러했다.

* Trajectory GRU는 이전 상태와 현재 입력을 결합하여 Local Information을 활용한다.
* ConvLSTM과 ConvGRU의 Invariant Filter와 다른 Variant 성격의 Filter를 가진다.
* 연결구조와 Filter Weight가 유연하다.
* 연산비용이 보다 최적화되었다.
* 다른 모델에 비해 높은 성능을 보여준다.



### 예측 결과

우리의 모델이 학습하여 최종적으로 테스트 데이터셋에 대해 추론한 결과는 다음과 같았다.

다음 그림은 $T-15$, $T-10$, $T-5$, $T$ 시점의 이미지를 Input으로 받은 모델이 $T+5$ 시점의 강우 이미지를 예측하는 것을 보여준다. 

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/160377932-600c23d3-2ca8-4c4d-8e20-5bcfbc19bc61.png">
</p>
<center><b>예측 반사도와 실제 반사도(빨강색 사각형)</b></center>
<br>

결과적으로 프로젝트의 결론은 이러했다.

* RainNet은 ConvLSTM2D 모델에 비해 학습비용 대비 약간 더 높은 정확도를 보여주었다. 다만, 이는 ConvLSTM 모델쪽의 학습조건이 불안정했기 때문에 확신할 수는 없었다.
* 앞선 네 시점의 레이더 기반 기상이미지를 이용하여 그 다음 시점의 이미지를 예측할 수 있었다.
* Trajectory GRU 모델, Time2Vec을 이용하여 더 높은 정확도의 개선을 기대할 수 있었지만, 시간관계상 모두 구현할 수는 없었다.

그리고 이렇게 눈으로 예측결과를 보는 것은 모델학습이 잘 되었는지에 대한 정량적인 판단의 근거로는 빈약했다. 실제로 Dacon에서 모델 성능을 평가받을 때는, Precipitation Nowcasting 분야의 Criterion이 적용되어 그 값을 기준으로 모델의 테스트 성능이 측정되었다.



## 프로젝트 결과

최종적으로 인공신경망을 활용한 강우예측 프로젝트, 댓글 오염도 분석 프로젝트 2개 모두 교육과정에서 우수작으로 선정되었다! 이때 완전 기분좋았다. ㅎㅎ

하지만 코로나 때문에.. "2020 강원 BigData Forum"에서 진행될 예정이었던 교육생 우수 프로젝트 시연회와 포스터 전시가 모두 취소되어 버렸다.

하지만! 알고보니 포럼 자체는 취소된 것이 아니었다. 당시에 코로나가 심해져서 50명 이상은 모일 수 없었는데, 그 때문에 학생들은 참석하지 않게 되고 일부 관계자들만 모여서 소소하게 행사가 진행되었던 것이다. 그때 찍힌 사진들 중에는 본 교육과정의 우수작으로 전시된 나의 작품 사진도 있었다! ㅎㅎㅎ

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/160247985-6cc468b0-cf27-4e4a-ab99-0af988de83ab.jpg">
</p>
<center><b>2020 강원 BigData Forum</b></center>
<br>

잘 보면.. 내 이름이 적힌 포스터가 2개 있다. ㅎㅎ

결과적으로 5개월 동안의 교육과정 + 한 달 동안의 프로젝트에 쏟은 노력이 빛을 발하게 되었다! 역시 노력은 배신하지 않는다. 그리고 6개월(840시간) 동안 해당 교육과정을 잘 이수했다는 수료증명서도 받았다. 완전 뿌듯 ㅎㅎ

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/176378925-5228b80a-6c66-40f0-b2b9-4c02e094d559.jpg" width=80%>
</p>
<center><b>교육과정 수료증!!</b></center>
<br>

교육과정을 수강한 시기는 대학원 석사 1학기였는데, 대학원 수업 및 연구를 진행하면서 동시에 교육과정을 수강하고 최종 프로젝트까지 진행하느라 정말 엄청나게 바빴다. 과장이 아니고, 아침 7시에 일어나 전날에 밀린 코딩 공부를 하고, 아침에 연구실에 출근하여 대학원 수업 및 연구를 진행하고, 저녁 이후부터 잠들기 전까지 다시 코딩을 하는 미친 일정이 6개월 동안 진행되었다!! 너무 힘들어서 중간에 대학원이나 교육과정 둘 중 하나를 포기해야 하나.. 진지하게 생각했다. 월 40만원의 교육수당이 없었다면 진작에 그만두었을지도 모르겠다.

여러가지로 힘든 점이 많았지만, 6개월 동안 정말로 많은 것을 배웠다. 그리고 최종 프로젝트를 진행하며 전에 배운 지식들을 실제로 활용해볼 기회가 있어서 더 좋았다.

교육과정 동안 친절하게 학생들을 이끌어주신 강원대학교 융합보안사업단의 이수안 선생님께 진심으로 감사를 드린다. 그리고 대학원 석사 첫 학기에 딴짓하는 제자를 관대하게 봐주셨던 지도교수 HSK 교수님께도 감사의 말씀을 드린다.

---
