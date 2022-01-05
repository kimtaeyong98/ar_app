# ar_app 어플리케이션
아루코 마커 위에 객체를 그리는 ar 어플리케이션 입니다.

3개의 객체를 각기 다른 아루코 마커를 인식해서 그 위에 그립니다.

카메라가 셋팅되어있어야하고, 카메라 실시간 스트리밍으로 아루코 마커를 찍어야합니다.

# 사전작업

1280x720(노트북 웹캠)을 사용하여 calibration을 진행


# 동작과정
1. 각 프레임마다 undistorted 진행하여 AR background image를 생성

2. 아루코 마커 인식 및 축 표시 (x,y,z)

3. 객체를 아루코 마커 중앙으로 이동 및 축 표시(x,y,z)

4. 객체를 좌우대칭으로 변경 후 애니메이션 (걷는 애니메이션 적용)

5. 랜더링 이펙트 적용

* marker1을 월드스페이스로 두고  marker2, marker3이 인식되면 해당 수식으로 마커 위에 그려지도록 구현


# 3개의 마커 트래킹
![image](https://user-images.githubusercontent.com/63800086/148220945-992b9139-2048-417e-97e7-e58b88ac71e8.png)


# 객체 중앙으로 이동
![image](https://user-images.githubusercontent.com/63800086/148220693-d6e66bcb-6d1b-4dce-856e-93261e58bf25.png)

# 객체 뒤집기
![image](https://user-images.githubusercontent.com/63800086/148220710-e924b56a-48a6-4ce4-93cd-57b150a672c8.png)

# 3개의 마커 3개의 객체
![image](https://user-images.githubusercontent.com/63800086/148220803-cc16c214-9f55-4f24-a8cd-67b136f4ff2a.png)

# 랜더링 이펙트
![image](https://user-images.githubusercontent.com/63800086/148220826-c70e7af1-f10f-4241-b6a7-5fdde3f2c1a1.png)

