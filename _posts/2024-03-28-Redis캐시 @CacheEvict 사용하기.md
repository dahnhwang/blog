---
title: "Redis캐시 @CacheEvict 사용하기"
layout: post
---

접속량이 높아 캐싱을 해둔 페이지인데, 업무로직에서 입력/수정/삭제 시마다 동일한 캐시네임(별도 파라미터가 없는 상태)을 모두 삭제하는 옵션설정(allEntries=true) 때문에 불필요하게 애써 캐싱해둔 내용이 사라지는 부분이 있었다. 비효율적인 부분이라 개선을 진행했다. 

스프링 cacheEvict 어노테이션에서 allEntries=true라고 설정할 경우, Redis에서는 KEYS라는 명령어를 통해 키 검색을 하게 된다. 이 때, 특정 캐시네임 뒤에 wild card(*) 파라미터가 오며 유사한 모든 키를 탐색하며 부하가 걸릴 수 있다.

{% highlight python %}
KEYS pattern
-- Returns all keys matching pattern.
{% endhighlight %}

<link href="/" rel="stylesheet">

Caching하는 쪽과 @CacheEvict하는 쪽 모두 동일한 구체적인 키값을 주는 방법으로 해결했다.

예를 들어, 특정 룸에 대한 리뷰가 업데이트된다면 해당 룸넘버와 룸의 리뷰개수에 대한 캐시를 모두 날려줘야하는 상황이다. 

{% highlight java %}
@Cacheable(value = "selectRoomNo", key = "{#p0.roomNo, #p1.roomType}")
public String selectRoomNo(Room room, Hotel hotel) {
	//do something...
}

@Cacheable(value = "selectReviewCount", key = "#p0.roomNo")
public String selectReviewCount(Room room) {
	//do something...
}

@Caching(evict = {
    @CacheEvict(value = "selectRoomNo", key = "{#p0.roomNo, #p1.roomType}"),
    @CacheEvict(value = "selectReviewCount", key = "#p0.roomNo")
})
public String updateRoomReview(Room room) {
	//do something...
}
{% endhighlight %}

만약 메소드의 파라미터가 없다면 0이라는 디폴트 값을 Key로 사용하여 저장한다. 그리고 만약 메소드의 파라미터가 여러 개라면 파라미터들의 hashCode 값을 조합하여 키를 생성한다.
하지만 여러 개의 파라미터 중에서도 1개의 키 값으로 지정하고 싶은 경우도 있다. 그러한 경우에는 위과 같이 Key 값을 별도로 지정해주면 된다.

이 때 주의할 점은 key가 String의 조합으로 생성된다는 점이다. 따라서 `selectRoomNo`의 key순서가 `#p0.roomNo, #p1.roomType`이고, `updateRoomReview`에서는 `#p1.roomType, #p0.roomNo`로 줬다면 둘은 동일한 문자열으로 생성되지 않으므로 evict시 정확히 원하는 캐시키를 타겟팅할 수 없게된다. 따라서 key부분의 순서를 맞춰주는 것이 중요하다.

한 가지 더, `selectRoomNo`의 `#p1.roomType = null`이고 `updateRoomReview`로 들어오는 파라미터 `#p1.roomType = ""`이라면 이 둘 역시 동일한 캐시키로 잡히지 않는다. 이유는 Serializer의 특성 때문으로 다음 포스팅에서 다루고자 한다.

참고
[레디스 공식문서](https://redis.io/commands/keys)
[Spring 의 CacheEvict 에서 allEntries=true 는 Redis에서 어떻게 동작하게 될까?](https://charsyam.wordpress.com/2022/04/18/%EC%9E%85-%EA%B0%9C%EB%B0%9C-spring-%EC%9D%98-cacheevict-%EC%97%90%EC%84%9C-allentriestrue-%EB%8A%94-redis%EC%97%90%EC%84%9C-%EC%96%B4%EB%96%BB%EA%B2%8C-%EB%8F%99%EC%9E%91%ED%95%98%EA%B2%8C/)
[캐시(Cache) 추상화와 사용법(@Cacheable, @CachePut, @CacheEvict)](https://mangkyu.tistory.com/179)

<font color='#909194'>Last updated: April 17, 2024</font>