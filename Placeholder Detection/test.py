from mlutils import model
import hashing_service, siamese_service
from hashing_service import HashingService
from siamese_service import SiameseService
from placeholder_service import PlaceholderService

import os
os.environ['http_proxy'] = "http://sysproxy.wal-mart.com:8080"
os.environ['https_proxy'] = "http://sysproxy.wal-mart.com:8080"

# gh = HashingService()
# gh.load()
# print(gh.predict('https://i5.walmartimages.com/asr/2d121697-0e9c-4c46-a9fb-6e85796a3f4b_1.07874315f91d9b41c3ddd3c63ec21a56.jpeg'))

# gh = SiameseService()
# gh.train()
# gh.save()
# gh.evaluate()
# print(gh.predict('https://i5.walmartimages.com/asr/dac87d61-f8bd-4802-9006-36283ab548c1_1.6ac60d844c34f15a58ffa100b7b50400.jpeg'))

gh = PlaceholderService()
# gh.train()
# gh.save()
# gh.evaluate()
urls = ['https://i5.walmartimages.com/asr/dac87d61-f8bd-4802-9006-36283ab548c1_1.6ac60d844c34f15a58ffa100b7b50400.jpeg', 'https://i5.walmartimages.com/asr/32533d16-a53b-486b-92be-69c986bf7373_1.d30e53b99c5f54c50274e163a638ae07.png', 'https://i5.walmartimages.com/asr/296343cc-2f81-4494-8e77-40f2f9811123_1.a68a28830f0151ab02c869640663ccd3.jpeg', 'https://i.walmartimages.com/i/p/_180X180.jpg']
print(gh.predict(urls))
