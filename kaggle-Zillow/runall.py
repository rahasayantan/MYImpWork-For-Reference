import os

print('Feature gen...')
#cmd = "python ./feature_gen_v4XX.py"
#os.system( cmd )

print('Keras Enc...')
cmd = "python ./keras_encode.py"
os.system( cmd )

print('Keras No outier..')
cmd = "python ./keras_wout.py"
os.system( cmd )

print('Keras with utlier...')
cmd = "python ./keras_with_outlier.py"
os.system( cmd )

print('LGB with Outier...')
#cmd = "python ./lgb_raw_wo.py"
#os.system( cmd )

print('XGB Count ENC...')
#cmd = "python ./xgb_count_enc_wo.py"
#os.system( cmd )

print('XGB Mean SHFT Enc...')
#cmd = "python ./xgb_meanshft_enc_wo.py"
#os.system( cmd )

print('RGF...')
#cmd = "python ./RGF.py"
#os.system( cmd )

print('ET...')
#cmd = "python ./et_woout.py"
#os.system( cmd )

print('Regression...')
#cmd = "python ./regression.py"
#os.system( cmd )

#print('Knn...') # too slow
#cmd = "python ./knn.py"
#os.system( cmd )


print('L2....')
#cmd = "python ./stacker.py"
#os.system( cmd )

