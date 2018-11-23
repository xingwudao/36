import xlearn as xl

param = {'task':'binary', 'lr':0.2,
         'epoch': 20, 'k':2,
         'lambda':0.002, 'metric':'auc'}

train_data = "../../data/criteo_conversion_logs/small_train.txt"
test_data = "../../data/criteo_conversion_logs/small_test.txt"

lr_model = xl.create_linear()
lr_model.setTrain(train_data)
lr_model.setValidate(test_data)
lr_model.setTest(test_data)
lr_model.setSigmoid()
lr_model.fit(param, './lr_model.out')

fm_model = xl.create_fm()
fm_model.setTrain(train_data)
fm_model.setValidate(test_data)
fm_model.setTest(test_data)
fm_model.setSigmoid()
fm_model.fit(param, './fm_model.out')

ffm_model = xl.create_ffm()
ffm_model.setTrain(train_data)
ffm_model.setValidate(test_data)
ffm_model.setTest(test_data)
ffm_model.setSigmoid()
ffm_model.fit(param, './ffm_model.out')
