import requests
count = 0
for i in range(20,23):
    for j in range(1,1000):
        if count == 20:
            count = 0
            break
        jtem = str(j)
        while len(jtem) < 4:
            jtem = "0"+jtem

        img_url = f"http://cb.dut.udn.vn/ImageSV/{i}/102{i}{jtem}.jpg"
        print(img_url)
        response = requests.get(img_url)
        if response.status_code != 404:
            count = 0
            fp = open(f'dataset/Image/Face/102{i}{jtem}.jpg', 'wb')
            fp.write(response.content)
            fp.close()
        else:
            count +=1