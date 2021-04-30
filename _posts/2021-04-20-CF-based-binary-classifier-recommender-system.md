---
title: "CF기반 Binary Classifier 추천모델 만들어보기"
layout: post
---

[Kaggle에 올라와있는 데이터셋](https://www.kaggle.com/kandij/job-recommendation-datasets)을 이용해 구직자의 채용공고 페이지 방문이력을 바탕으로 특정 채용공고에 지원할지 안할지를 예측하는 Collaborative Filtering 기반 Binary Classifier 추천모델을 만들어본다.



## Dataset
총 5개의 데이터셋 중 구직자의 채용공고 페이지 방문이력 정보가 들어있는 `Job_Views.csv` 중 구직자 식별정보(`Applicant.ID`), 채용 식별정보(`Job.ID`) 및 페이지 확인일시(`View.Start`) 컬럼 만을 이용해볼 것이다.  
{% highlight python %}
raw[['Applicant.ID','Job.ID','View.Start']].head(3)
{% endhighlight %}

| Applicant.ID          | Job.ID          | View.Start         |
|------------------|------------------|-----------------|
| 10000     | 73666     | 2014-12-12 20:12:35 UTC     |
| 10000	      | 96655      | 	2014-12-12 20:08:50 UTC   |
| 10001		      | 84141      | 	2014-12-12 20:12:32 UTC   |

{% highlight python %}
user_count = raw['Applicant.ID'].unique().shape[0]
job_count = raw['Job.ID'].unique().shape[0]
print(user_count, job_count) 
{% endhighlight %}

3448명의 구직자와 7047개의 채용공고로 이루어진 테이블이다. 



## Data Preprocessing Planning
이 실험에서는 페이지 확인일시(View.Start)값이 있는 경우 구직자의 관심있음을 뜻하는 1, 값이 없는 경우 0으로 하는 `checked` 컬럼을 새로 만들어 진행해볼 것이다. 크게 아래 순서로 진행한다.

1. 식별자 레이블링 작업
2. 페이지 확인일시(`View.Start`)를 1의 값을 가지는 새로운 컬럼 `checked`로 변환
3. 페이지 확인일시(`View.Start`)의 관계가 없는(즉 `checked`가 0인 경우) 구직자-채용정보 간 테이블을 생성하여 기존 테이블과 합침

위 1,2,3의 결과로 페이지 확인여부(`checked`)를 매개로 하는 구직자-채용정보 테이블을 만들 수 있으며 아래와 같은 결과예시의 데이터를 모델학습을 위한 Input으로 사용할 것이다.

| Applicant.ID          | Job.ID          | checked         |
|------------------|------------------|-----------------|
| 1     | 1     | 1     |
| 1	      | 2      | 	1   |
| 2		      | 3      | 	0   |



## Data Preprocessing


### 1. 식별자 레이블링 작업

{% highlight python %}
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder

user_enc = LabelEncoder()
raw['Applicant.ID'] = user_enc.fit_transform(raw['Applicant.ID'].values)

item_enc = LabelEncoder()
raw['Job.ID'] = item_enc.fit_transform(raw['Job.ID'].values)
{% endhighlight %}

sklearn의 LabelEncoder를 사용하여 10000부터 매겨져있는 기존 테이블의 ID를 정수 1부터 시작하는 것으로 새롭게 라벨링을 했다. 데이터셋에서 사용자의 ID가 정수로 되어있는 경우는 드물기 때문에 무조건 처리해주고 시작하는게 좋은 것 같다.


### 2. 페이지 확인일시(`View.Start`)를 1의 값을 가지는 새로운 컬럼 `checked`로 변환

{% highlight python %}
count = raw['View.Start'].isnull().sum()
count 
{% endhighlight %}

페이지 확인일시(`View.Start`) 정보가 없는 행이 하나도 없기 때문에 이를 새로운 컬럼으로 변환하는 작업을 계속 진행한다.

{% highlight python %}
raw['checked'] = 1
raw = raw.drop(['View.Start'], axis=1)
raw.head(3)
{% endhighlight %}

| Applicant.ID          | Job.ID          | checked         |
|------------------|------------------|-----------------|
| 980     | 1610     | 1     |
| 980	      | 1610      | 	1   |
| 981		      | 2989      | 	1   |

그런데 위 테이블에서 볼 수 있듯이 한 명의 구직자가 동일한 채용공고를 여러번 확인했을 수 있으므로 중복되는 행은 제외하도록 한다.

{% highlight python %}
train.drop_duplicates(subset=None, keep="first", inplace=True)
{% endhighlight %}

이렇게 하여 `checked`의 값이 1인 총 ㅍ 된 테이블이 되었다.


### 3. 페이지 확인일시(`View.Start`)의 관계가 없는(즉 `checked`가 0인 경우) 구직자-채용정보 간 테이블을 생성하여 기존 테이블과 합침

주어진 데이터셋의 경우 구직자가 채용공고를 확인한 데이터만 존재하고 그렇지 않은 경우의 데이터는 존재하지 않는다. 따라서 확인하지 않은 데이터를 새로 생성하기로 하였다.

{% highlight python %}
user_uid = raw['Applicant.ID'].unique()
job_uid = raw['Job.ID'].unique()
user_uid_table = DataFrame({'Applicant.ID':user_uid})
job_uid_table = DataFrame({'Job.ID':job_uid})
{% endhighlight %}

구직자 및 채용공고의 unique 값으로 이루어진 각각의 다른 두 개의 테이블 `user_uid_table` 와 `job_uid_table`을 생성해 이 둘을 곱집합(Cartesian product)한 테이블을 만들 것이다. 즉, 모든 구직자에 대한 모든 채용공고의 관계(`full_table`)를 나타낼 것이다. 

{% highlight python %}
user_uid_table['checked'] = 0
job_uid_table['checked'] = 0
{% endhighlight %}

{% highlight python %}
user_uid_table.head(3)
{% endhighlight %}

| Applicant.ID          | checked          |
|------------------|------------------|
| 980     | 0     |
| 981	      | 0      |
| 982		      | 0      |

{% highlight python %}
job_uid_table.head(3)
{% endhighlight %}

| Job.ID          | checked          |
|------------------|------------------|
| 1610     | 0     |
| 1611	      | 0      |
| 1612		      | 0      |

위와 같은 3448 × 2짜리 `user_uid_table`과 7047 × 2짜리 `job_uid_table`을 `checked`를 통해 조인시켜 `full_table`을 만든다.

{% highlight python %}
full_table = user_uid_table.merge(job_uid_table, on='checked') # 24298056 rows 
{% endhighlight %}

결과로 3,448 x 7,047의 결과값인 24,298,056개의 행을 가진 테이블이 생성되었다. 이제 기존의 `raw` 테이블과 `full_table`을 concat을 통해 합칠 것인데, `Applicant.ID`와 `Job.ID`가 중복되는 경우 `checked`값이 1인 row를 그대로 보존하여 모델에 입력할 최종 데이터셋을 만들 것이다.

{% highlight python %}
concat_df = pd.concat([full_table, raw]) 
concat_df
{% endhighlight %}

중복제거 전 테이블의 행끼리 그냥 합친 `concat_df`는 24306976 rows × 3 columns로 확인된다.

{% highlight python %}
train = concat_df.drop_duplicates(subset=['Applicant.ID', 'Job.ID'], keep='last')
{% endhighlight %}

학습에 사용할 `train` 테이블은 `concat_df`의 행 수(24,306,976) - `raw`의 행 수(8,920)인 24,298,056개의 행으로 확인되어 중복처리가 잘 이루어졌음을 확인할 수 있다.

{% highlight python %}
train
{% endhighlight %}

| Applicant.ID          | Job.ID          |    checked      |
|------------------|------------------|-----------------|
| 980     | 2298     | 0    |
| 980	      | 1899      | 	0   |
| 980		      | 1425      | 	0   |
| 980		      | 487      | 	0   |
| 980		      | 4230      | 	0   |
| ...		      | ...      | 	...   |
| 979		      | 1404      |1   |
| 979		      | 17      | 	1   |
| 979		      | 54      | 	1   |
| 979		      | 2      | 	1   |
| 979		      |52      | 	1   |

{% highlight python %}
count = train['checked'].value_counts()
{% endhighlight %}

구직자와 채용공고의 관계 중 `checked`가 0인 것은 24,289,136개, 1인 것은 본래 `raw` 데이터셋에서도 확인하였듯이 8,920개로 0의 분포가 약 99.9%의 비율로 나타난다. 이러한 비균형적인 데이터 분포에 대해 처리해주는 기법이 존재하지만 금번 실험에서는 우선 제외하고 진행해보았다.



## Train/Validation/Test set splitting

테스트셋은 원칙적으로는 처음부터 분리를 했어야 하나, 여기서는 한번에 Train, Validation, Test셋을 분리하였다. 20%의 Test셋을 분리한 다음 나머지 80%를 다시 8:2의 비율로 나눠 학습과 검증으로 사용하였다.

{% highlight python %}
X = train[['Applicant.ID', 'Job.ID']].values
y = train['checked'].values.astype('float32')

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=34)

X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, stratify=y_train, random_state=34)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
{% endhighlight %}

학습과 검증으로 사용될 데이터셋의 형태 및 수는 다음과 같다.
((15550755, 2), (3887689, 2), (15550755,), (3887689,))



## Model Training

Collaborative Filtering 중에서 Matrix Factorization을 사용해 사용자(구직자)와 아이템(채용공고)의 Latent Factor 행렬을 학습시켜볼 것이다.

{% highlight python %}
# modeling imports
from keras.models import Model
from keras.layers import Add, Activation
from keras.layers import Input, Reshape, Dot
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras import optimizers
from keras.optimizers import RMSprop
{% endhighlight %}

케라스를 이용해 모델링을 할 것이므로 필요한 모듈을 모두 import해둔다.

{% highlight python %}
class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors
    
    def __call__(self, x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=regularizers.l2(1e-4))(x)
        x = Reshape((self.n_factors,))(x)
        return x
{% endhighlight %}

우선 축소된 차원의 구직자/채용공고 벡터를 리턴하는 클래스 `EmbeddingLayer`를 정의한다. `EmbeddingLayer` 클래스에서는 Keras에서 제공하는 Embedding 함수를 사용하여 구직자/채용공고의 벡터 차원(`n_items`)를 `n_factors`크기의 벡터로 변환한다.

예를 들어 구직자 벡터만을 놓고 설명하면, 데이터셋에 총 3명의 구직자가 있다면 1번 구직자의 벡터는 [1,0,0], 2번 구직자의 벡터는 [0,1,0], 3번 구직자의 벡터는 [0,0,1]이라는 3차원의 원핫 인코딩으로 나타낼 수 있다. 이를 3차원보다 작은 `n_factors` 사이즈로 축소(=데이터 안에서 `n_factors` 개수만큼 구직자를 그룹으로 분류)하게 되는데, (채용공고 벡터와 Dot product를 통해) 가장 실제값과 유사한 예측값을 내놓게되는 `n_factors` 차원의 구직자 벡터를 찾는 것이 학습의 목표다. 

{% highlight python %}
def Recommender(n_users, n_items, n_factors):
    user = Input(shape=(1,))
    u = EmbeddingLayer(n_users, n_factors)(user)
    ub = EmbeddingLayer(n_users, 1)(user)
    
    item = Input(shape=(1,))
    i = EmbeddingLayer(n_items, n_factors)(item)
    ib = EmbeddingLayer(n_items, 1)(item) 
    
    x = Dot(axes=1)([u, i])
    x = Add()([x, ub, ib])
    y = Activation('sigmoid')(x)

    model = Model(inputs=[user, item], outputs=y)
    rms = RMSprop(learning_rate=0.1)
    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
    return model
{% endhighlight %}

이해를 돕기 위해 아래와 같이 도식도를 그려보았다.

![CF_model](/blog/assets/CF_model.png)

`n_users`명의 구직자와 `n_items`개의 채용공고가 동일한 `n_factors` 차원의 벡터로 각각 임베딩된 뒤 내적(Dot product)이 행해지고, 활성화함수인 sigmoid를 거쳐 결과로 나온 하나의 single value가 바로 특정 채용공고에 대한 구직자의 지원 여부 예측값(확률)이 된다. 정답이 있는 지도학습이므로 예측값이 실제값과 얼마나 다른지 판단하는 binary cross entropy라는 손실함수를 통해 다시 네트워크(임베딩 벡터)를 반복적으로 학습시킨다.

Binary Classification 문제의 경우 활성화함수로 sigmoid function을 사용하는데, 이해를 위해서 찾고 또 찾다가 한 [유튜브 강의](https://www.youtube.com/watch?v=WsFasV46KgQ)에서 명쾌한 해답을 얻었다. 참고하면 좋다.


{% highlight python %}
n_factors = 100
X_train_array = [X_train[:, 0], X_train[:, 1]]
X_val_array = [X_val[:, 0], X_val[:, 1]]

model = Recommender(user_count, job_count, n_factors)
{% endhighlight %}

잠재인수 `n_factors`를 100차원으로 놓고 모델객체를 생성하였다. 차원이 커질 수록 학습속도는 더 느려지고 오버피팅될 확률이 크다. 여러번 해봤지만 100정도가 적당했던 것 같다.

{% highlight python %}
history = model.fit(x=X_train_array, y=y_train, batch_size=10000, epochs=5, verbose=1, validation_data=(X_val_array, y_val))
{% endhighlight %}

`batch_size` 및 `epoch`를 각각 [1000, 5000, 10000], [5, 10, 20] 씩 변화시켜가며 학습시켜보았는데 크게 차이가 없었다. 아래는 가장 빠른 조건으로 학습을 시켜본 결과.

![CF_loss](/blog/assets/CF_loss.png)
![CF_accuracy](/blog/assets/CF_accuracy.png)

<font color='#909194'>Last updated: April 25, 2021</font>