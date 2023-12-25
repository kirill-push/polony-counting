# Changelog

<!--next-version-placeholder-->

## v0.6.0 (2023-12-25)

### Feature

* :sparkles: add func to calculate T7c phages from input (mock) ([`0904c7c`](https://github.com/kirill-push/polony-counting/commit/0904c7c47381bb280c58de6aa63bf0ce72f83678))
* :sparkles: add func to calculate T7 phages from input ([`0c7102f`](https://github.com/kirill-push/polony-counting/commit/0c7102f7df389c02d85f23426e1934a13f532f17))
* :sparkles: add func to calculate T4 phages from input ([`fe5b50c`](https://github.com/kirill-push/polony-counting/commit/fe5b50c3dc8dc880c40291e786c9f5dba64d7a61))
* :sparkles: add func to convert polony count to phage ([`dd4dc19`](https://github.com/kirill-push/polony-counting/commit/dd4dc19af683e1ab05570526901fd6bdcc1b848f))
* :sparkles: add new func to save predictions to csv file ([`d030352`](https://github.com/kirill-push/polony-counting/commit/d03035242381f332a0f571be521dbbd2cbd148a8))
* :sparkles: add to predict function classification ([`1cfc268`](https://github.com/kirill-push/polony-counting/commit/1cfc268014f6be51ce07d6c0ae901fb8f79276e2))
* :sparkles: add classifier model to predict_one_image func ([`0bf5f62`](https://github.com/kirill-push/polony-counting/commit/0bf5f627f99ef34e4bacc1caa88ff43886a16dc4))

### Documentation

* :memo: add table of contents to readme ([`bced160`](https://github.com/kirill-push/polony-counting/commit/bced16058436de744dc477ab0b9078c90c7450c2))
* :memo: update readme ([`d08b816`](https://github.com/kirill-push/polony-counting/commit/d08b8160c565543c9b116d00470900436c9e46af))
* :memo: update main documentation page ([`17c4d7e`](https://github.com/kirill-push/polony-counting/commit/17c4d7e5a47d7c512cdb7d1d9a515ac72452f7af))
* :memo: update readme according to researchers recommendations ([`e031d05`](https://github.com/kirill-push/polony-counting/commit/e031d0599dbb771cfe64481207b25ea90d6b89b4))
* :memo: update project structure in readme ([`3b5dcd9`](https://github.com/kirill-push/polony-counting/commit/3b5dcd90d57fd256cf64c7c896f78fd8a013c14d))
* :memo: update projects structure in readme ([`2f9f2ff`](https://github.com/kirill-push/polony-counting/commit/2f9f2ff15b1425472bb0aea7214630f781f3d0d2))

## v0.5.0 (2023-12-02)

### Feature

* :sparkles: add new method pos_weight to PolonyDataset ([`1eb3e2a`](https://github.com/kirill-push/polony-counting/commit/1eb3e2af46aa407f87a143ab0fc199678edba400))
* :sparkles: add new mode to Looper for training classifier ([`b6cb927`](https://github.com/kirill-push/polony-counting/commit/b6cb927aeace9af477f9d79dfa41d1b47aa51f77))
* :sparkles: add new mode to train classifier ([`9f8b94f`](https://github.com/kirill-push/polony-counting/commit/9f8b94f3a8be04d9a48d1ca0b0825a5d1d6fb6b5))
* :sparkles: new mode for PolonyDataset ([`420d2e9`](https://github.com/kirill-push/polony-counting/commit/420d2e918bbecada2581c454fae6780c4a695bef))
* :sparkles: new mode for count data func ([`62d91e3`](https://github.com/kirill-push/polony-counting/commit/62d91e396cebdc30b7902e2d2ea9d4488545acd0))
* :sparkles: new working mode for generate squares ([`3dd71fb`](https://github.com/kirill-push/polony-counting/commit/3dd71fb821e56f2cac0fbcf23f14a01a781950d9))
* :sparkles: new working mode for generate data ([`40962ed`](https://github.com/kirill-push/polony-counting/commit/40962edd0239177711fee8012c1cc2c5fd8fcd6b))
* :sparkles: new working mode for creating empth hdf5 ([`d640a2d`](https://github.com/kirill-push/polony-counting/commit/d640a2d78b3149ddebb1ce87efd7253e82877dea))
* :sparkles: add squares classifier model ([`c508e4d`](https://github.com/kirill-push/polony-counting/commit/c508e4d1d2ea1304153495895d6f28df79176523))

## v0.4.0 (2023-11-26)

### Feature

* :sparkles: add default train function parameters from config ([`f9b81e9`](https://github.com/kirill-push/polony-counting/commit/f9b81e94f40d827e8e6b49bc677c22ab3eb3992f))

### Fix

* :bug: correct indexing in path dict from dataset ([`e2fadb6`](https://github.com/kirill-push/polony-counting/commit/e2fadb61f5718cb8ae2071934ab20498e1ea40d5))
* :wrench: update learning_rate format to decimal in training config ([`83f9061`](https://github.com/kirill-push/polony-counting/commit/83f9061cfafa48945c7a11b224c5fe6c25da9f4f))
* :bug: correct string-based indexing in path dictionary ([`16056d3`](https://github.com/kirill-push/polony-counting/commit/16056d3a7dd8bf9939ac3b2c8a3ed4455353d3e2))
* :pencil2: fix typos in dataset config ([`2f58d6f`](https://github.com/kirill-push/polony-counting/commit/2f58d6f01d173aa10986be7661a286d46dc6e491))
* :art: change format of dataset_path param in PolonyDataset ([`4f0be85`](https://github.com/kirill-push/polony-counting/commit/4f0be85174d2475a54a9cd23b9316c35539b37e4))
* :bug: remove deprecated parameter from train ([`c0757b1`](https://github.com/kirill-push/polony-counting/commit/c0757b1ea4c910b74333611649e15377fa86f6b0))
* :art: change train import in init ([`4cb9ba8`](https://github.com/kirill-push/polony-counting/commit/4cb9ba81e1d3297d6e6508e0fb8b0befa035c8aa))
* :sparkles: update reading of sys json with paths to images ([`d558798`](https://github.com/kirill-push/polony-counting/commit/d558798473b58aa8f25dfec1a3a413eb50860c04))
* :bug: change json path for correct work of package ([`1b2bbd3`](https://github.com/kirill-push/polony-counting/commit/1b2bbd3cf7b4931e456f771150510f8133f74dc6))
* :zap: change standart path to save data after download ([`e9cddda`](https://github.com/kirill-push/polony-counting/commit/e9cddda5765aef6616bec207056fb4240255041f))
* :bug: change path for list of images with errors ([`03c75b0`](https://github.com/kirill-push/polony-counting/commit/03c75b006404738b4ec2d72d5024b473a7fbd6bc))
* :truck: change config path ([`2ed478e`](https://github.com/kirill-push/polony-counting/commit/2ed478ed1dbd30f0ad1faf2cce5bbd6840501ce6))
* :pushpin: change numpy version to resolve bug ([`a6caf24`](https://github.com/kirill-push/polony-counting/commit/a6caf247a5f903dff68f640296d35f8e65266df1))

## v0.3.1 (2023-11-22)

### Fix

* :bricks: update imports and paths for new layout ([`33b4dac`](https://github.com/kirill-push/polony-counting/commit/33b4dac7a13a1ec4d6c2eeb3d3d769faf1a66900))
* :building_construction: change project layout ([`d4deb1a`](https://github.com/kirill-push/polony-counting/commit/d4deb1a02e14107b93e3383f27e3f09f2485c7ac))

### Documentation

* :bookmark: add project version ([`9dafb26`](https://github.com/kirill-push/polony-counting/commit/9dafb261c42aea2481e42e066899e81860a8e5f2))
* :memo: update project layout in readme ([`5bc2308`](https://github.com/kirill-push/polony-counting/commit/5bc2308c405cb04971c4d480cc9b65ace69b08a9))

## v0.3.0 (2023-11-20)

### Feature

* :sparkles: add script execution functionality in main program mode ([`712046a`](https://github.com/kirill-push/polony-counting/commit/712046a7d7a507a979d8fb5a484eed20f724622c))
* :sparkles: add new func to fix bugs ([`fcd05a9`](https://github.com/kirill-push/polony-counting/commit/fcd05a9797859488e8bc95bf0446289e4a3e2fc0))

### Fix

* :bug: add missing params to dataset class ([`7ef116d`](https://github.com/kirill-push/polony-counting/commit/7ef116d9d62aa8da1f25839a7661751a0da85e40))
* :bug: import error fix ([`b9dd1c8`](https://github.com/kirill-push/polony-counting/commit/b9dd1c8a04396bb06864ad48267dac25d0e76905))
* :bug: fix bug after new test ([`70d5cef`](https://github.com/kirill-push/polony-counting/commit/70d5cefe9369d97d157ee730fad74a80738be9d1))
* :bug: fix bugs after new tests ([`58614b3`](https://github.com/kirill-push/polony-counting/commit/58614b39b578f1544ed706d7685f9bab7c2bb0f9))

### Documentation

* :memo: add coverage badge to README ([`55bcfed`](https://github.com/kirill-push/polony-counting/commit/55bcfed7f41ab5ae120cb77a299aed56d83a8a47))
* :memo: add documentation badge to readme ([`ea9d8f5`](https://github.com/kirill-push/polony-counting/commit/ea9d8f5e99bb64de5affb787b125c81752c7282d))
* :memo: add release badge to readme ([`f901f1a`](https://github.com/kirill-push/polony-counting/commit/f901f1a4712c85aa9974559c2ce4d6889b0ab90d))
* :memo: update CHANGELOG.md ([`b623660`](https://github.com/kirill-push/polony-counting/commit/b6236604c3d24b11cae5c2c5239540e98a00076b))

## v0.2.1 (2023-11-14)

### Fix

* :bug: experiments with testing workflow to fix bug ([`bbe6934`](https://github.com/kirill-push/polony-counting/commit/bbe69345e84d83471b5bdb3a5760f853590b4253))
* :bug: bug fix ([`b8c896e`](https://github.com/kirill-push/polony-counting/commit/b8c896e9aa7828c7330b77871225518beeb0a219))

## v0.2.0 (2023-11-14)

### Feature

* Add releases ([`2a14aa5`](https://github.com/kirill-push/polony-counting/commit/2a14aa50d285340c5f564ac0c22d620431ad00dc))

##

### Feature
* add lint to workflow ([PR](https://github.com/kirill-push/polony-counting/pull/37))
* add testing to workflow ([PR](https://github.com/kirill-push/polony-counting/pull/38))
* add documentation and project site ([PR](https://github.com/kirill-push/polony-counting/pull/40))

### Add
* add tests to fully cover model.py ([PR](https://github.com/kirill-push/polony-counting/pull/39))
