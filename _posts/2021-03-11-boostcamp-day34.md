---
layout: post
title: "Day34. Panoptic segmentation, Landmark localization, GAN"
subtitle: "Instance/Panoptic segmentation, Landmark localization, Conditional generative model, GAN"
date: 2021-03-11 23:59:12+0900
background: '/img/posts/bg-posts.png'
use_math: true
---

## 개요 <!-- omit in toc -->
> 오늘은 semantic segmentation에서 더욱 진보된 형태인 Instance/Panoptic segmentation, 그리고 또 다른 CV task인 landmark localization에 대해서 다루었다. 마지막으로 GAN에 대해서도 살펴보았다. 어제도 그렇고, 현재 다루고 있는 내용들은 사실 짧은 시간 내에 모두 깊게 살펴보기에는 어려운 부분이 많다. 그래서 큰 그림을 먼저 이해하고 세세한 내용은 추후 다시 살펴보는 것이 바람직할 것 같다.  
      
이 글은 아래와 같은 내용으로 구성된다.  
- [Instance Segmentation](#instance-segmentation)
    - [Mask R-CNN](#mask-r-cnn)
    - [YOLACT / YolactEdge](#yolact--yolactedge)
- [Panoptic Segmentation](#panoptic-segmentation)
    - [UPSNet](#upsnet)
    - [VPSNet](#vpsnet)
- [Landmark Localization](#landmark-localization)
    - [Hourglass network](#hourglass-network)
    - [DensePose](#densepose)
    - [RetinaFace / More extension](#retinaface--more-extension)
- [Detecting objects as Keypoints](#detecting-objects-as-keypoints)
    - [CornerNet](#cornernet)
    - [CenterNet](#centernet)
- [Conditional Generative Model](#conditional-generative-model)
- [Reference](#reference)
    
<br />
  
## Instance Segmentation
instance segmentation은 기존 semantic segmentation의 task에 **distinguishing instances**, 즉 인스턴스별 레이블이 다르게 매겨지는 것까지를 원한다. 
어떻게 보면 비슷한 task를 수행하므로 여기서는 기존 semantic segmentation에 쓰였던 Faster R-CNN과 YOLO 등의 모델을 확장하여 활용하게 된다.    
  
<br />

#### Mask R-CNN  
먼저 Mask R-CNN을 살펴보자. Mask R-CNN은 Faster R-CNN과 거의 똑같다. 다만 거기에 **mask branch**라는 새로운 단이 추가된다.  
  
![mask_r_cnn](/img/posts/34-1.png){: width="90%" height="90%"}{: .center}   
또 다른 차이점으로, RoI Pooling 대신 **RoIAlign**이라는 기법이 사용된다. 기존 RoI Pooling은 RoI가 소수점 좌표를 가지고 있을 경우 반올림하여 Pooling을 해준다는 특징이 있다. (딱 grid 단위로 쪼개서 본다)
즉, RoI Pooling은 정수 좌표만을 처리할 수 있다. 이러한 처리는 classification 처리에서는 문제가 없지만 segmentation task에서는 위치가 왜곡되기 때문에 문제가 발생한다.  
  
![roi_align](/img/posts/34-2.png){: width="90%" height="90%"}{: .center}   
Mask R-CNN에서는 RoIAlign이라는 기법을 활용한다. 현재 문제는 정수 좌표만 볼 수 있다는 점이다.
RoIAlign은 정수 좌표(즉, 점선으로 되어있는 grid 좌표)를 가지고 bilinear interpolation을 한다. 
최종적으로 **구하고자 하는 점에 해당하는 feature 값**을 구할 수 있게 된다. 
이 방법을 통해 보다 정교한 feature 값을 뽑아낼 수 있게 되어 큰 성능향상을 보였다고 한다.  
  
한편, 맨 위에 첨부한 이미지를 보면 class/box regression head 외에 아래쪽에 새로운 **mask branch**가 도입되었다. 
이 부분에서는 각 class별로 binary mask prediction을 수행한다. 먼저 upsampling을 한 후 클래스 수만큼의 mask(여기서는 80개)를 모조리 생성한 후 
위쪽 head에서 해당 이미지의 classification을 완료하면 이를 참조하여 그 class에 해당하는 mask를 최종적으로 출력하게 된다.  
  
그 외에도 RPN이전 feature map 추출 단게에서 FPN 구조를 활용하여 전후의 정보를 모두 고려해주었다는 특징이 있다.
이렇게 U-Net, FPN과 같은 구조는 이후에도 게속해서 쓰이는데 이 구조가 아무래도 전체 context를 고려할 수 있어 성능이 보장되는 면이 있는 것 같다.  
  
<br />

#### YOLACT / YolactEdge
YOLACT는 이름에서 보이듯이 YOLO를 확장한 모델이다.   
![YOLACT](/img/posts/34-3.png){: width="90%" height="90%"}{: .center}  
여기서도 FPN 구조를 활용하였고 이번엔 **Protonet**을 도입하여 mask의 prototype을 추출한다. 
prototype이란, 추후에 마스크를 만들 수 있는 재료를 뜻한다. 선형대수학으로 보면 mask space를 span하는 basis가 여기에 해당한다고 보면 될 것 같다. 
결국 protonet 단에서는 이 basis를 추출한다.    
  
이제 basis를 추출했으면 앞에 붙는 계수(coefficient)도 있어야 mask를 만들 수 있다. **Prediction head**단에서는 각 detection에 대하여
protoype 합성을 위한 계수를 출력해낸다. 최종적으로 bounding box에 맞추어 이를 선형결합(weighted sum)한 후 Crop/Threshold를 거쳐 mask response map을 생성한다.   
  
여기서의 핵심은 역시 Prototype 단이다. Mask R-CNN은 각 클래스별 mask를 모두 뽑아낸다. 이렇게 되면 메모리에 부담이 있을 수 있는데 YOLACT는 prototype을 미리 설정해
저장해야하는 mask의 수를 최소화한다는 특징이 있다.  

![YolactEdge](/img/posts/34-4.png){: width="90%" height="90%"}{: .center}  
또 다른 모델로 **YolactEdge**가 있다. YOLACT는 빠르지만 경량화된 모델은 아니라서 작은 device에서 활용하기 어렵다. 
YolactEdge는 위와 같이 YOLACT에서 바로 이전 time의 keyframe의 FPN에서 추출된 특징을 현재 time에서도 재활용하여 계산량을 최소화하였다.  
  
하지만 아직도 video에서 완전한 real time task를 수행하기에는 한계점이 많아 비디오에서의 실시간 처리는 아직까지 연구중인 분야라고 한다.  
  
<br />

## Panoptic Segmentation
**panoptic segmentation**은 instance는 물론이고 배경의 segment까지 추출할 수 있는 모델이다. 
semantic segmentation은 배경을 인식할 수 있지만 같은 클래스의 instance를 구별해내지 못한다. 
instance segmentation에서는 배경을 추출할 수 없다. panoptic segmentation은 따라서 이 둘의 특성을 모두 활용한다.  
  
<br />

#### UPSNet
![UPSNet](/img/posts/34-5.png){: width="90%" height="90%"}{: .center}  
UPSNet은 panoptic segmentation을 위해 2019년에 제시된 모델이다. 시기만 봐도 이 분야의 개척이 시작된지 얼마 되지 않았음을 알 수 있다. :worried:  
  
역시 FPN 구조를 앞단에서 활용하고 뒷단에서는 semantic head와 instance head를 각각 두어 윗단에서는 semantic segmenation을 수행하는 FCN처럼 conv 연산을 수행하여
segment prediction을 한다. 아랫단에서는 물체에 대한 detection을 통해 mask logit을 찾는다. 최종적으로 이 두 정보를 함께 활용하여 panoptic logits를 도출할 수 있다.  
  
![UPSNet_instance_panoptic_head](/img/posts/34-6.png){: width="90%" height="90%"}{: .center}  
두 정보를 통합하여 이용하는 panoptic head쪽을 좀 더 자세히 살펴보자.   
  
instance head에서 흘러온 feature는 원래 이미지의 원래 위치에 넣어주기 위해 resize/pad를 한다. 
semantic head에서 흘러온 정보는 먼저 semantic의 배경 부분을 나타내는 mask를 바로 output으로 흘려보내고($x\_{\text{stuff}}$), 
물체부분이 마스킹된 feature($x\_{\text{thing}}$)의 경우 위쪽으로는 instance head의 원래 위치를 맞춰주기 위해 instance head의 출력과 함께 활용되고, 아래 쪽으로는 전체 feature에서 해당 물체에 대한 mask를 제거하기 위해 활용된다. 이걸 제거해서 생긴 부분은 unknown class로 활용된다.   
  
최종적으로 이들 결과를 모두 concatenation하여 이를 panoptic logits으로 활용한다.  
  
<br />

#### VPSNet
![VPSNet](/img/posts/34-7.png){: width="90%" height="90%"}{: .center}  
panoptic segmentation을 video에서 하기 위한 모델로 **VPSNet**이 있다. 여기서는 **motion map**을 활용한다.
motion map은 시간의 흐름에 따라 물체가 어디로 가는지, 즉 매 frame별 각 점들이 일대일로 대응되는 대응점들을 나타낸 map이다. (물론 새로 나타난 물체는 새로운 점이 필요하다)  
  
프로세스는 다음과 같다. (1) 이전 시점($t - \tau$) RoI feature들을 motion map을 통해 현재 시점($t$)에서 tracking하고 (2) 현재 시점 $t$에서 또 따로 FPN을 통해
찾아낸 feature를 (1)에서 구한 것과 concatenation한 후 이를 통해 최종적으로 현재 시점의 RoI feature를 추출한다. (3) 이전 시점 RoI feature와 현재 시점 RoI feature를 
Track head의 input으로 주어 최종적으로 두 시점의 RoI간 연관성을 추출해내고 마지막으로 그 뒷단에서는 UPSNet과 같은 작업을 수행한다.  
  
VPSNet은 track head를 통해 같은 물체는 같은 id를 가지도록 시간에 따른 tracking을 수행한다.  
  
<br />

## Landmark Localization
**landmark localization**은 keypoint estimation이라고도 불리는데, 한마디로 주어진 이미지의 주요 특징 부분(landmark, keypoint)를 추정/추적하는 task를 말한다. 
여기서 landmark라 함은 사람의 얼굴의 눈, 코, 입이라거나 사람 전신의 전체 뼈대(skeleton) 등을 생각해볼 수 있다. 
물론 landmark를 무엇으로 정하느냐는 어떤 이미지에 대한 작업인지에 따라 다르며 이는 모델을 짜는 사람이 미리 정해야하는 hyperparameter라고 볼 수 있다.  
  
![landmark_localization](/img/posts/34-8.png){: width="90%" height="90%"}{: .center}  
이를 위한 방법으로 **coordinate regression**을 먼저 생각해볼 수 있다. 모든 landmark 후보 각각에 대하여 각각의 x, y 좌표를 예측하는 방법이다. 
하지만 이건 좀 부정확하고 generalization에 있어 문제가 있다. 
여기서는 대신 **heatmap classification**을 다뤄보고자 한다. heatmap classification은 coordinate regression보다 성능이 더 우월하다.   
  
heatmap classification의 최종 결과값은 각 채널이 사전 정의한 landmark 각각에 대한 위치별 확률값을 나타내게된다. 이전에 본 semantic/instance segmentation에서 수행했던 작업과 유사하다. 
결국 여기서의 차이점은 keypoint 각각을 하나의 class로 생각하고 그걸 기반으로 각 keypoint별 heatmap을 찍어주는 것 뿐이다. 다만 이 방법은 모든 픽셀에 대한 확률을 그려야 하므로 
computational cost가 높다는 단점이 있다.   
  
heatmap classification을 위해서는 결국 landmark가 어떤 좌표 (x, y)에 있는지를 우선적으로 찾아야한다. 
그런데 딱 그 위치에서 ground truth(확률값 label)가 1이고 그 주변은 모두 0이면 학습이 제대로 안 될 것이다.
그래서 **먼저 각 landmark가 존재하는 실제 위치 (x, y)를 가지고 그 주변에 heatmap label을 만들어야한다.**  
  
이를 위해 여기서는 **Gaussian heatmap**을 다뤄보도록 한다.   
  
![Gaussian_heatmap](/img/posts/34-9.png){: width="50%" height="50%"}{: .center}  
방법은 간단하다. 해당 물체가 있는 실제 위치 $(x\_c, y\_c)$를 기준으로 주변 좌표들에 Gaussian 분포를 적용하면 된다. 그러면 위와 같이 confidence 값에 대한 **heatmap label**을 만들 수 있다.  
  
식은 아래와 같다. 

<center>

$$
G_\sigma(x, y) = \exp \left(-\frac{(x-x_c)^2 + (y-y_c)^2}{2\sigma^2} \right)
$$

</center>
  
$\sigma$는 hyperparameter로 heatmap label을 얼마나 퍼뜨려줄지 직접 정해야한다. 
실제 구현에서 x좌표와 y좌표는 그 dimension이 다르지만 <code>numpy</code>에서 broadcasting을 적용하므로 최종적으로 2차원의 heatmap을 얻을 수 있다. 
그렇다면 **Gaussian heatmap에서 landmark location으로의 변환은 어떻게 할 수 있을까?** 지금 당장 내 생각에는 Gaussian heatmap에서 gradient가 0인 극점의 좌표를 찾으면 
그곳이 landmark의 실제 위치일 것 같은데 일단 이 부분은 숙제로 남겨두도록 한다.  

<br />

#### Hourglass network
![hourglass_network](/img/posts/34-10.png){: width="90%" height="90%"}{: .center}  
**Hourglass network**에서는 stack된 구조가 모래시계(hourglass)처럼 생겼다고 해서 붙여진 이름이다. 
모래시계 구조는 역시 UNet 구조와 매우 유사하게 생겼다. 다만 여기서는 그런 구조를 여러번 stack하였다. 
이렇게 하여 row level의 정보와 high level의 정보를 모두 인식하고, 이를 반복함으로써 성능을 더욱 개선한다. 
또한 맨 앞단에서는 영상 전체에서 feature map을 미리 추출하여 각 픽셀이 receptive field를 더욱 크게 가져갈 수 있도록 해주었다.  

![hourglass](/img/posts/34-11.png){: width="50%" height="50%"}{: .center}  
**hourglass 모양 부분(stack)**은 위와 같이 UNet과 매우 유사하다.
차이점을 위주로 보자면 (1) skip connection을 할 때 conv 연산을 해주고 전달한다. UNet에서는 이런 과정 없이 바로 전달하였다.  
  
또한 (2) concatenation 대신 sum을 활용하였다. 이렇게 하면 dimension이 늘어나지 않는 특징이 있다.  
  
구조를 보면 UNet보다는 사실상 FPN에 가까운 구조이다. 아무튼 이런 피라미드 구조가 landmark localization task에서도 역시 잘 동작한다.  

<br />

#### DensePose
![densepose](/img/posts/34-12.png){: width="80%" height="80%"}{: .center}  
**DensePose**는 기존 2D 좌표만을 예측하였던 모델에서 벗어나 **3D 구조까지** 예측한다. 
위와 같이 3D pose estimation에 유용하다. 
여기서는 **UV map**을 활용한다. UV map이란 쉽게 말해 3D 구조를 2D에 표현한 map을 의미한다. 
이 map상에서는 물체의 위치가 시간에 따라 변화해도 그 좌표가 불변한다는 성질이 있다.  
    
![densepose_architecture](/img/posts/34-13.png){: width="80%" height="80%"}{: .center}  
DensePose는 Mask R-CNN에서처럼 Faster R-CNN에 **3D surface regression branch**라는 새로운 브랜치를 얹는다. 
위와 같이 각 body part에 대한 segmentation map을 추출해낼 수 있으며, 2D 구조의 CNN으로 3D 위치까지 예측할 수 있다는 데에 의의가 있다.  

<br />

#### RetinaFace / More extension
![retina_face](/img/posts/34-14.png){: width="80%" height="80%"}{: .center}    
이후 나온 **RetinaFace**라는 모델에서도 FPN backbone에 task 수행을 위한 head를 얹는다. 
특이한 점은, **multi-task 수행을 위한 여러 branch를 모두 얹는다는 것**이다.   
  
위 그림을 보면 현재 사람의 얼굴에 대한 semantic segmentation, landmark regression, 3D vertices regression 등 여러 문제를 해결할 수 있는 헤드를 한꺼번에 얹었다. 
물론 각 헤드가 수행하는 역할은 다르지만, **근본적으로 모든 것이 '사람의 얼굴'이라는 같은 대상에 대한 학습을 하므로 공통된 정보에 대한 역전파가 가능하다는 점**이 이 모델의 핵심 아이디어다.  
  
이렇게 하면 동시에 모든 task를 수행할 수 있는 동시에, backbone network가 더욱 강하게 학습된다. 사실상 **데이터를 더 많이 본 효과**를 낼 수 있다.
따라서 이러한 모델은 적은 데이터로도 보다 robust한 model을 만들 수 있게 된다.   
  
여기서 깨달을 수 있는 점은, **backbone은 계속 재활용/공용하더라도 target task를 위한 head만 새로 설계하면 다양한 응용이 가능하다는 것이다.** 
이러한 기조가 최근 CV 분야의 큰 디자인 패턴 흐름 중 하나이다.  
  
<br />

## Detecting objects as Keypoints  
object detection을 할 때 bounding box가 아니라 keypoint 형태로도 detection을 할 수 있다. 
결국 하고자하는 것은 같긴 한데, RoI를 먼저 찾거나 할 필요 없이 그냥 keypoint를 찾아서 그 keypoint를 기준으로 무언가 작업을 하면 object detection이 가능하다. 
여기서는 이러한 작업을 수행했던 CornetNet과 CenterNet에 대해 아주 간단히 알아보고 넘어가도록 하자.  
  
<br />

#### CornerNet
![corner_net](/img/posts/34-15.png){: width="80%" height="80%"}{: .center}  
**CornetNet**은 좌측상단점과 우측하단점을 탐색하는 모델이다. 
이를 위해 이미지를 backbone에 통과시키고 거기서 뽑아낸 피쳐맵을 **총 4개의 head에 통과시킨다.**   
  
헤드 하나하나의 역할을 살펴보자면, (1) top-left corner 위치를 찾는 헤드, (2) 그것에 대한 embedding을 뽑아내는 헤드, (3) bottom-right corner 위치를 찾는 헤드, (4) 그것에 대한 embedding을 뽑아내는 헤드로 구성된다.   
  
여기서 embedding head는 corner 점의 embedding을 뽑아내는데, embedding은 위 그림과 같이 각 점이 표현하는 정보를 나타내며, **같은 물체에 대한 점들은 같은 embedding을 나타내도록 학습**된다. 
따라서 각 점의 embedding의 결과를 참조하여 우리는 물체의 위치를 점 2개로 detection 할 수 있다. (점 2개로도 unique한 bounding box를 결정할 수 있다) 
이 모델은 정확도보다는 속도에 무게를 주었다. 따라서 정확도는 좀 떨어지는 면이 있다.  

<br />
  
#### CenterNet
**CenterNet**에서는 CornetNet을 개선하여 좌상, 우하에 더불어 중점까지도 탐색한다. 
center point를 추가로 도입함으로써 총 6개의 head가 필요하겠지만 이로 인해 정확도는 향상되었다.   
  
![center_net](/img/posts/34-16.png){: width="80%" height="80%"}{: .center}   
더 진보된 CenterNet은 아예 중심점만을 찾고, 거기에 추가적으로 높이(height), 너비(width) 값을 찾는다. 
이렇게 함으로써 얻는 장점은 무엇일까? 정확한 것은 논문을 읽어봐야 알겠지만, 직관적으로 먼저 생각을 해보자.   
  
앞서 본 첫번째 CenterNet은 정확도가 향상되는 대신 head가 6개가 필요하였다. 
CornetNet에서는 정확도는 떨어지지만 head를 4개만 두어 속도를 취했다. 지금 보고있는 진보된 CenterNet은 CornerNet처럼 head를 4개만 두어 속도를 취하는 한편
동시에 이전 CenterNet과 같은 작업을 수행하므로 둘 모두의 장점을 취했다고 이해할 수 있다.   
  
이렇게 만들어진 CenterNet은, 논문에서 Faster R-CNN/Retina Net/YOLOv3의 object detection보다도 성능 및 속도면에서 우월했음을 제시하고 있다.  

<br />

## Conditional Generative Model   
**작성중**  
  
<br />

## Reference   
[Mask R-CNN](https://ganghee-lee.tistory.com/40)  
[Mask R-CNN(2)](https://cdm98.tistory.com/33)  