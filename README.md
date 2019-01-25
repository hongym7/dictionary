### 요즘 Sigmoid 보다 ReLU를 많이 쓰는데 그 이유는?
  Non-Linearity라는 말의 의미와 그 필요성은?  
  ReLU로 어떻게 곡선 함수를 근사하나?  
  ReLU의 문제점은?  
  Bias는 왜 있는걸까?

A) Neural Network에서 신경망이 깊어질수록 학습이 어렵기 때문에, 전체 레이어를 한번 계산한 후, 그 계산 값을 재활용 하여, 다시 계산하는 Back propagation이라는 방법을 사용하는데, sigmoid 함수를 activation 함수로 사용할 경우, 레이어가 깊어지면 back propagation이 제대로 작동하지 않기 때문에,(값을 뒤에서 앞으로 전달할 때 희석이 되는 현상. 이를 Gradient Vanishing 이라고 한다) ReLu를 사용

+ ReLU의 문제점 : 입력값이 <0 일시, 함수 미분값이 0이 되는 약점이 있음.
	PReLU : ReLU의 입력값이 <0 일때의 약점을 보완함 (상대적으로 인기가 없음… 왜?)
+ Bias는 왜 있을 걸까?
	Bias는 학습 데이터가 가중치와 계산되어 넘어야 하는 임계점으로, 임계점 값이 높으면 높을수록 분류의 기준이 엄격하다는 것을 의미힌다. 따라서 Bias가 높을수록 모델이 간단해지는 경향이 있으며 오히려 underfitting의 위험이 발생하게 된다. 반대로 Bias가 낮을수록 임계점이 낮아 데이터의 허용 범위가 넓어지는 만큼 학습 데이터에만 잘 들어맞는 모델이 만들어질 수 있다(overfitting).






2. Gradient Descent에 대해서 쉽게 설명한다면?
A) 딥러닝에서 목적은 최적의 매개변수를 학습 시에 찾아야 한다. 여기에서 최적이란 손실 함수가 최솟값이 될 때의 매개변수이다. 그러나 매개변수 공간이 광대하여 어디가 최솟값이 되는 곳인지 알아내기는 쉽지 않다. 이런 상황에서 기울기를 잘 이용해 함수의 최솟값 (또는 가능한 한 작은 값)을 찾으려는 것이 경사법 (경사하강법) 이다.
+ 왜 꼭 Gradient를 써야 할까?
+ 그 그래프에서 가로축과 세로축 각각은 무엇인가?
	세로축 : Cost, 가로축 : Weight Value
+ 실제 상황에서는 그 그래프가 어떻게 그려질까?
	y=x^2 모양의 그래프가 아닌 다양한 곡선 모양을 가진 그래프가 그려진다.
+ GD 중에 때때로 Loss가 증가하는 이유는?
	
+ 중학생이 이해할 수 있게 더 쉽게 설명 한다면?
+ Back Propagation에 대해서 쉽게 설명 한다면?










3. Local Minima 문제에도 불구하고 딥러닝이 잘 되는 이유는?
A) weight와 bais를 잘 초기화 시키는 방법(pre-training by RBM, dA)가 제시되었음 (이전에는 random하게 initialize 하였음) pre-training을 통해서 효과적으로 layer를 쌓아서 여러 개의 hidden layer도 효율적으로 훈련시킬 수 있게 되었기 때문에 딥러닝이 잘 되고 있음
+ GD가 Local Minima 문제를 피하는 방법은?
	Momentum을 적용한다. 관성을 이용하여, 학습 속도를 더 빠르게 하고, 변곡점을 잘 넘어갈 수 있도록 해주는 역할을 수행함.
+ 찾은 해가 Global Minimum인지 아닌지 알 수 있는 방법은?
	잘 모르겠음… 다른 자료에 의하면 minimum 이라고 하는 것은 현재 차원에서 이동할 수 있는 모든 방향으로 gradient 값이 증가 하는 방향이어야 하는데, 이런 경우가 희박하다고 설명
	DNN과 같은 고차원 구조에서는 대부분 local minima가 아니라 saddle point일 가능성이 높음.
	실제 local minima가 존재한다면 그것은 global minimum과 거의 차이가 없을 가능성이 높음(neural network의 대칭성)










4. CNN에 대해서 아는대로 얘기하라
A) Convolutional Neural Network. 합성곱 신경망. 이미지 인식분야에서 좋은 결과를 내고 있으며 음성인식이나 자연어 처리에서도 좋은 효과를 낸다. CNN 모델은 기본적으로 Convolution Layer Pooling Layer로 구성된다. 이 계층들을 얼마나 많이, 또 어떠한 방식으로 쌓느냐에 따라 성능 차이는 물론 풀 수 있는 문제가 달라질 수 있다.
+ CNN이 MLP보다 좋은 이유는?
	MLP(Multi Layer Perceptron). MLP는 모든 입력이 위치와 상관없이 동일한 수준의 중요도를 갖는다고 본다. 그렇기 때문에 이를 이용해 fully-connected neural network를 구성하게 되면 파라미터의 크기가 엄청나게 커지는 문제가 생긴다. 예를 들어 입력층이 28x28개라고 했을 때, 784개의 가중치를 찾아내야 하지만, 컨볼루션 계층에서는는 3x3 9개의 가중치만 찾아내면 된다.
+ 어떤 CNN의 파라메터 개수를 계산해 본다면?
+ 주어진 CNN과 똑같은 MLP를 만들 수 있나?
	가능하다.
+ 풀링시에 만약 Max를 사용한다면 그 이유는? 
	영역이 windows에서 가장 큰 자극을 선택
+ 시퀀스 데이터에 CNN을 적용하는 것이 가능할까?








5. Word2Vec의 원리는?

+ 그 그림에서 왼쪽 파라메터들을 임베딩으로 쓰는 이유는?
+ 그 그림에서 오른쪽 파라메터들의 의미는 무엇일까?
+ 남자와 여자가 가까울까? 남자와 자동차가 가까울까?
+ 번역을 Unsupervised로 할 수 있을까?

















6. Auto Encoder에 대해서 아는대로 얘기하라
A) 비지도 학습(unsupervised learning) 중 가장 널리 쓰이는 신경망. 오토인코더는 입력값과 출력값을 같게 하는 신경망이며, 가운데 계층의 노드 수가 입력값도 적은 것이 특징이다. 이러 구조로 인해 입력 데이터를 압축하는 효과를 얻게 되고, 또 이 과정이 노이즈 제거에 매우 효과적이라고 알려졋다. 오토인코더의 핵심은 입력층으로 들어온 데이터를 인코더를 통해 은닉층으로 내보내고 은닉층의 데이터를 디코더를 통해 출력층으로 내보낸 뒤, 만들어진 출력값을 입력값과 비슷해지도록 만드는 가중치를 찾아내는 것이다. 
+ MNIST AE를 TF나 Keras등으로 만든다면 몇줄일까?
	Tensorflow : 약 40 line / Keras : ???
	질문의 의도를 정확히 모르겠음.
+ MNIST에 대해서 임베딩 차원을 1로 해도 학습이 될까?
+ 임베딩 차원을 늘렸을 때의 장단점은?
+ AE 학습시 항상 Loss를 0으로 만들 수 있을까?
	입력층과 출력층이 같다면 Loss 가 0이 되는 것이 아닌가?
	(좀 더 찾아봐야 할 듯)
+ VAE는 무엇인가?
	 Variational AutoEncoder(VAE). (내용추가)








7. Training 세트와 Test 세트를 분리하는 이유는?
	학습시킨 모델이 새로 들어오는 입력에 대해 결과를 어느 정도 수준으로 예측할 수 있는지 판단할 수 있는 근거로 Test 세트를 사용한다. Training data set만 있을 경우에는 같은 데이터에 대해서는 100% 정확도로 예측을 할 수 있지만 새롭게 들어오는 데이터에 대해서는 어느 정도 수준으로 예측이 가능할지 알 수 없다.
출처 : http://pythonkim.tistory.com/24
+ Validation 세트가 따로 있는 이유는?
	validation set을 사용하는 이유는 간단합니다. 바로 "모델의 성능을 평가하기 위해서" 입니다.  training을 한 후에 만들어진 최종 모형이 잘 예측을 하는지 그 성능을 평가하기 위해서 사용합니다. training set의 일부를 모델의 성능을 평가하기 위해서 희생하는 것입니다. 하지만 이 희생을 감수하지 못할만큼 data set의 크기가 작다면 cross-validation이라는 방법을 쓰기도 합니다. cross-validation은 training set을 k-fold 방식을 통해 쪼개서 모든 데이터를 training과 validation 과정에 사용할 수 있게 합니다. 

출처: http://3months.tistory.com/118
+ Test 세트가 오염되었다는 말의 뜻은?
	Test 세트에 Train 세트에 포함된 데이터가 있는 경우를 의미하는 것 같음
출처 : https://tensorflow.blog/tag/urbansound8k/
+ Regularization이란 무엇인가?
	모델의 일반화 오류를 줄여 과적합을 방지하는 기법을 총칭
- L1, L2 regularization이 일반적인 기법, Loss 뒤에 weight에 대한 패널티 텀을 부여한다
- Dropout 기법 : Forward-pass시 랜덤하게 뉴런의 연결을 끊어버리는 기법으로 보통 0.5의 확률로 dropout을 적용
출처 : http://astralworld58.tistory.com/64
      http://sonjju.tistory.com/248


8. Batch Normalization의 효과는?
A) Batch Normalization은 기본적으로 Gradient Vanishing / Gradient Exploding 이 일어나지 않도록 하는 아이디어 중의 하나이다. 
기존 Deep Network에서는 learning rate를 너무 높게 잡을 경우 gradient가 explode/vanish 하거나, 나쁜 local minima에 빠지는 문제가 있었다. 이는 parameter들의 scale 때문인데, Batch Normalization을 사용할 경우 propagation 할 때 parameter의 scale에 영향을 받지 않게 된다. 따라서, learning rate를 크게 잡을 수 있게 되고 이는 빠른 학습을 가능케 한다.
Batch Normalization의 경우 자체적인 regularization 효과가 있다. 이는 기존에 사용하던 weight regularization term 등을 제외할 수 있게 하며, 나아가 Dropout을 제외할 수 있게 한다 (Dropout의 효과와 Batch Normalization의 효과가 같기 때문.) . Dropout의 경우 효과는 좋지만 학습 속도가 다소 느려진다는 단점이 있는데, 이를 제거함으로서 학습 속도도 향상된다.
https://shuuki4.wordpress.com/2016/01/13/batch-normalization-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84/
+ Dropout의 효과는?
+ BN 적용해서 학습 이후 실제 사용시에 주의할 점은? 코드로는?
	주의해야할 점 하나는 train 과정에서는 mini-batch의 sample mean/variance를 사용하여 BN transform을 계산하였지만, inference를 할 때에도 같은 규칙을 적용하게 되면 mini-batch 세팅에 따라 inference가 변할 수도 있기 때문에 각각의 test example마다 deterministic한 결과를 얻기 위하여 sample mean/variance 대신 그 동안 저장해둔 sample mean/variance들을 사용하여 unbiased mean/variance estimator를 계산하여 이를 BN transform에 이용한다.
http://sanghyukchun.github.io/88/
+ GAN에서 Generator 쪽에도 BN을 적용해도 될까?	
	다만 모든 layer에 BN을 추가하면 문제가 생기고 Generator의 output layer와 discriminator의 input layer에는 BN을 넣지 않는다고 세세한 팁이 있습니다. 이유를 추측해보자면 아무래도 Generator가 생성하는 이미지가 BN을 지나면서 다시 normalize 되면 아무래도 실제 이미지와는 값의 범위가 다를 수 밖에 없으니 그런 문제가 생기지 않을까 싶습니다. BN을 넣으면 GAN의 고질적 문제 중 하나인 Mode collapsing 문제를 어느 정도 완화해준다고 하는데 이 부분은 사실 아직 해결되었다고 보기는 어렵습니다
	http://jaejunyoo.blogspot.com/2017/02/deep-convolutional-gan-dcgan-1.html

9. SGD, Momentum, RMSprop, Adam에 대해서 아는대로 설명한다면?
출처 : http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html
A) Momentum : Momentum 방식은 말 그대로 Gradient Descent를 통해 이동하는 과정에 일종의 ‘관성’을 주는 것이다. 현재 Gradient를 통해 이동하는 방향과는 별개로, 과거에 이동했던 방식을 기억하면서 그 방향으로 일정 정도를 추가적으로 이동하는 방식이다.

+ SGD에서 Stochastic의 의미는?
	Stochastic(확률론적) Gradient Descent (SGD) 라는 방법을 사용한다. 이 방법에서는 loss function을 계산할 때 전체 데이터(batch) 대신 일부 조그마한 데이터의 모음(mini-batch)에 대해서만 loss function을 계산한다. 이 방법은 batch gradient descent 보다 다소 부정확할 수는 있지만, 훨씬 계산 속도가 빠르기 때문에 같은 시간에 더 많은 step을 갈 수 있으며 여러 번 반복할 경우 보통 batch의 결과와 유사한 결과로 수렴한다. 또한, SGD를 사용할 경우 Batch Gradient Descent에서 빠질 local minima에 빠지지 않고 더 좋은 방향으로 수렴할 가능성도 있다.
+ 미니배치를 작게 할 때의 장단점은?
+ 모멘텀의 수식을 적어 본다면?
	vt=γvt−1+η∇θJ(θ)
	θ=θ−vt








10. 간단한 MNIST 분류기를 MLP+CPU 버전으로 numpy로 만든다면 몇줄일까?

+ 어느 정도 돌아가는 녀석을 작성하기까지 몇시간 정도 걸릴까?
+ Back Propagation은 몇줄인가?
+ CNN으로 바꾼다면 얼마나 추가될까?
11. 간단한 MNIST 분류기를 TF나 Keras 등으로 작성하는데 몇시간이 필요한가?

+ CNN이 아닌 MLP로 해도 잘 될까?
+ 마지막 레이어 부분에 대해서 설명 한다면?
+ 학습은 BCE loss로 하되 상황을 MSE loss로 보고 싶다면?
+ 만약 한글 (인쇄물) OCR을 만든다면 데이터 수집은 어떻게 할 수 있을까?

12. 간단한 MNIST DCGAN을 작성한다면 TF 등으로 몇줄 정도 될까?

+ GAN의 Loss를 적어보면? 
+ D를 학습할 때 G의 Weight을 고정해야 한다. 방법은?
+ 학습이 잘 안 될 때 시도해 볼 수 있는 방법들은?

13. 딥러닝할 때 GPU를 쓰면 좋은 이유는?
A) GPU는 직렬 처리를 하는 CPU와 다르게 (CPU도 코어의 개수만큼 병렬처리가 가능하나, GPU가 수행할 수 있는 병렬처리 능력에 미치지 못함) 수 많은 연산을 병렬처리가 가능하다. DL의 대부분 계산은 행렬 연산이고 이러한 행렬 연산을 병렬처리로 연산을 수행하게 되면 더 빠른 처리를 할 수 있게 된다. 또한 GPU는 CPU에 비해 더 높은 메모리 대역폭을 가지고 있기 때문에 큰 데이터에 대한 처리가 용이하다.
+ 학습 중인데 GPU를 100% 사용하지 않고 있다. 이유는?
	GPU의 메모리가 부족해서 GPU 연산 능력을 100% 사용하지 못함. 이때 가장 손쉬운 해결 방법으로는 배치 사이즈를 줄여서 메모리 사용량을 줄이는 것이다.
+ GPU를 두개 다 쓰고 싶다. 방법은?
	
+ 학습시 필요한 GPU 메모리는 어떻게 계산하는가?
	

14. TF 또는 Keras 등을 사용할 때 디버깅 노하우는?

15. Collaborative Filtering에 대해 설명한다면?

16. AutoML이 뭐하는 걸까?
