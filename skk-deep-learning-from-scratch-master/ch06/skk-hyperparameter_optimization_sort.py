# coding: utf-8
import numpy as np
results_val = {}

# ================================================
weight_decay = 10 ** np.random.uniform(-8, -4)
lr = 10 ** np.random.uniform(-6, -2)
val_acc_list = [0.56, 0.67, 0.87]
print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
results_val[key] = val_acc_list 
# ================================================
weight_decay = 10 ** np.random.uniform(-8, -4)
lr = 10 ** np.random.uniform(-6, -2)
val_acc_list = [0.35, 0.45, 0.96]
print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
results_val[key] = val_acc_list
# ================================================
weight_decay = 10 ** np.random.uniform(-8, -4)
lr = 10 ** np.random.uniform(-6, -2)
val_acc_list = [0.61, 0.73, 0.91]
print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
results_val[key] = val_acc_list

i=0;
for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[0][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)
    i=i+1
  
#val acc:0.87 | lr:4.888747353951389e-05, weight decay:5.46839547751622e-05
#val acc:0.96 | lr:0.0001409014984181345, weight decay:6.422170675854728e-08
#val acc:0.91 | lr:0.0024690602317237673, weight decay:2.2345631272665062e-07
#Best-1(val acc:0.96) | lr:0.0001409014984181345, weight decay:6.422170675854728e-08
#Best-2(val acc:0.91) | lr:0.0024690602317237673, weight decay:2.2345631272665062e-07
#Best-3(val acc:0.87) | lr:4.888747353951389e-05, weight decay:5.46839547751622e-05

for key, val_acc_list in results_val.items() :
    print("key: " + key + "  val_acc_list:", val_acc_list)

#key: lr:1.619149...32e-06, weight decay:3.831150818352792e-08  val_acc_list: [0.56, 0.67, 0.87]
#key: lr:0.001016...819493, weight decay:3.961339611017048e-08  val_acc_list: [0.35, 0.45, 0.96]
#key: lr:0.001722...419841, weight decay:9.844989358507206e-06  val_acc_list: [0.61, 0.73, 0.91]

    