from mlutils import model
import hashing_service, siamese_service
from hashing_service import HashingService
from siamese_service import SiameseService
from placeholder_service import PlaceholderService

import os
os.environ['http_proxy'] = "http://sysproxy.wal-mart.com:8080"
os.environ['https_proxy'] = "http://sysproxy.wal-mart.com:8080"

# gh = HashingService()
# gh.train()
# gh.save()
# gh.evaluate()

# gh = SiameseService()
# gh.train()
# gh.save()
# gh.evaluate()

gh = PlaceholderService()
request = [
  {
    "product_type": 'Area Rugs',
    "secondaryURL": [
      'https://i5.walmartimages.com/asr/32533d16-a53b-486b-92be-69c986bf7373_1.d30e53b99c5f54c50274e163a638ae07.png', 'https://i5.walmartimages.com/asr/296343cc-2f81-4494-8e77-40f2f9811123_1.a68a28830f0151ab02c869640663ccd3.jpeg'
    ],
    "primaryURL": [
      'https://i5.walmartimages.com/asr/dac87d61-f8bd-4802-9006-36283ab548c1_1.6ac60d844c34f15a58ffa100b7b50400.jpeg'
    ],
    "product_id": "29YGQUXYBJSG",
    "item_id": "213049978"
  }
]
print(gh.predict(request))
