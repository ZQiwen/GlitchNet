import os
from rbm import RBM
from au import AutoEncoder
import tensorflow as tf
import input_seismo
import matplotlib.pyplot as plt

# presettings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 
                    r'data', 
                    'Directory for storing data')
flags.DEFINE_integer('epochs', 50, 'The number of training epochs')
flags.DEFINE_integer('epochs_au', 50, 'The number of training epochs for the autoencoder')
flags.DEFINE_integer('batchsize', 64, 'The batch size')
flags.DEFINE_boolean('restore_rbm', False, 'Whether to restore the RBM weights or not.')

# ensure output dir exists
if not os.path.isdir('out'):
  os.mkdir('out')

mydataset = input_seismo.read_data_sets(FLAGS.data_dir, sp_data=True)

trX, trY, teX, teY = mydataset.train.seismos, mydataset.train.labels, mydataset.test.seismos, mydataset.test.labels
# trX, teX = min_max_scale(trX, teX)

npt=1200

# RBMs
rbmobject1 = RBM(npt, 256, ['rbmw1', 'rbvb1', 'rbmhb1', 'rbmva1', 'rbmha1'])
rbmobject2 = RBM(256, 128, ['rbmw2', 'rbvb2', 'rbmhb2', 'rbmva2', 'rbmha2'])
rbmobject3 = RBM(128, 64,  ['rbmw3', 'rbvb3', 'rbmhb3', 'rbmva3', 'rbmha3'])
rbmobject4 = RBM(64,  32,  ['rbmw4', 'rbvb4', 'rbmhb4', 'rbmva4', 'rbmha4'])


if FLAGS.restore_rbm:
  rbmobject1.restore_weights('./out/rbmw1.chp')
  rbmobject2.restore_weights('./out/rbmw2.chp')
  rbmobject3.restore_weights('./out/rbmw3.chp')
  rbmobject4.restore_weights('./out/rbmw4.chp')

# Autoencoder
autoencoder = AutoEncoder(npt, [256, 128, 64, 32], [['rbmw1', 'rbmhb1', 'rbmha1'],
                                                     ['rbmw2', 'rbmhb2', 'rbmha2'],
                                                     ['rbmw3', 'rbmhb3', 'rbmha3'],
                                                     ['rbmw4', 'rbmhb4', 'rbmha4']], tied_weights=False)

iterations = int(len(trX) / FLAGS.batchsize)

# Train First RBM
print('first rbm')
for i in range(FLAGS.epochs):
  for j in range(iterations):
    batch_xs, batch_ys = mydataset.train.next_batch(FLAGS.batchsize)
    rbmobject1.partial_fit(batch_xs)
  print('ipoch=%d\t cost=%f\t mean(va)=%f\t mean(ha)=%f' % (i,
                                            rbmobject1.compute_cost(teX),
                                            rbmobject1.o_va.mean(),
                                            rbmobject1.o_ha.mean()))
  # show_image("out/1rbm.jpg", rbmobject1.n_w, (28, 28), (30, 30))
rbmobject1.save_weights('./out/rbmw1.chp')

# Train Second RBM2
print('second rbm')
for i in range(FLAGS.epochs):
  for j in range(iterations):
    batch_xs, batch_ys = mydataset.train.next_batch(FLAGS.batchsize)
    # Transform features with first rbm for second rbm
    batch_xs = rbmobject1.transform(batch_xs)
    rbmobject2.partial_fit(batch_xs)
  print('ipoch=%d\t cost=%f\t mean(va)=%f\t mean(ha)=%f' % (i,
                                          rbmobject2.compute_cost(rbmobject1.transform(teX)),
                                          rbmobject2.n_va.mean(),
                                          rbmobject2.n_ha.mean()))
  # show_image("out/2rbm.jpg", rbmobject2.n_w, (30, 30), (25, 20))
rbmobject2.save_weights('./out/rbmw2.chp')

# Train Third RBM
print('third rbm')
for i in range(FLAGS.epochs):
  for j in range(iterations):
    # Transform features
    batch_xs, batch_ys = mydataset.train.next_batch(FLAGS.batchsize)
    batch_xs = rbmobject1.transform(batch_xs)
    batch_xs = rbmobject2.transform(batch_xs)
    rbmobject3.partial_fit(batch_xs)
  print('ipoch=%d\t cost=%f\t mean(va)=%f\t mean(ha)=%f' % (i,
                                          rbmobject3.compute_cost(rbmobject2.transform(rbmobject1.transform(teX))),
                                          rbmobject3.n_va.mean(),
                                          rbmobject3.n_ha.mean()))
  # show_image("out/3rbm.jpg", rbmobject3.n_w, (25, 20), (25, 10))
rbmobject3.save_weights('./out/rbmw3.chp')

# Train Forth RBM
print('fourth rbm')
for i in range(FLAGS.epochs):
  for j in range(iterations):
    batch_xs, batch_ys = mydataset.train.next_batch(FLAGS.batchsize)
    # Transform features
    batch_xs = rbmobject1.transform(batch_xs)
    batch_xs = rbmobject2.transform(batch_xs)
    batch_xs = rbmobject3.transform(batch_xs)
    rbmobject4.partial_fit(batch_xs)
  print('ipoch=%d\t cost=%f\t mean(va)=%f\t mean(ha)=%f' % (i,
                                          rbmobject4.compute_cost(rbmobject3.transform(rbmobject2.transform(rbmobject1.transform(teX)))),
                                          rbmobject4.n_va.mean(),
                                          rbmobject4.n_ha.mean()))
rbmobject4.save_weights('./out/rbmw4.chp')


# Load RBM weights to Autoencoder
autoencoder.load_rbm_weights('./out/rbmw1.chp', ['rbmw1', 'rbmhb1', 'rbmha1'], 0)
autoencoder.load_rbm_weights('./out/rbmw2.chp', ['rbmw2', 'rbmhb2', 'rbmha2'], 1)
autoencoder.load_rbm_weights('./out/rbmw3.chp', ['rbmw3', 'rbmhb3', 'rbmha3'], 2)
autoencoder.load_rbm_weights('./out/rbmw4.chp', ['rbmw4', 'rbmhb4', 'rbmha4'], 3)

# Train Autoencoder
print('autoencoder')
for i in range(FLAGS.epochs_au):
  cost = 0.0
  for j in range(iterations):
    batch_xs, batch_ys = mydataset.train.next_batch(FLAGS.batchsize)
    cost += autoencoder.partial_fit(batch_xs)
  print(cost)

autoencoder.save_weights('./out/au.chp')
autoencoder.load_weights('./out/au.chp')

fig, ax = plt.subplots()

print(autoencoder.transform(teX)[:, 0])
print(autoencoder.transform(teX)[:, 1])


plt.scatter(autoencoder.transform(teX)[:, 0], autoencoder.transform(teX)[:, 1], alpha=0.5)
plt.show()

# raw_input("Press Enter to continue...")
plt.savefig('out/myfig')

if 0:
  
  # some figrues to show the performance of autoencoder on seismograms in test 
  # dataset
  fig, ax = plt.subplots()
  import numpy as np
  teX_rec=autoencoder.reconstruct(teX)
  relative_E_error=np.divide(np.sum(np.square(teX_rec-teX),axis=1), np.sum(np.square(teX),axis=1))
  selected=np.where(relative_E_error<1)[0][:10]
  fig, ax = plt.subplots(figsize=(4,12))
  kk=0
  for itrace in selected:
    plt.plot(np.linspace(-5,25,npt),teX[itrace,:]+kk*2,'k')
    plt.plot(np.linspace(-5,25,npt),teX_rec[itrace,:]+kk*2,'r')
    kk+=1
  plt.show()
  fig, ax = plt.subplots()
  axis1,axis2 = 1,0
  
  noise=np.random.normal(0,0.1,teX.shape)
  sine_func=np.sin(np.linspace(-5,25,npt)).reshape(1,npt)
  square_func=np.ones(shape=(1,npt))
  square_func[0,int(npt/2):]=0
  triang_func=square_func-np.linspace(1,0,npt)
  plt.scatter(autoencoder.transform(noise)[:, axis1], autoencoder.transform(noise)[:, axis2], alpha=0.1,color='k')
  plt.scatter(autoencoder.transform(sine_func)[:, axis1], autoencoder.transform(sine_func)[:, axis2], alpha=1,color='b',s=60)
  plt.scatter(autoencoder.transform(square_func)[:, axis1], autoencoder.transform(square_func)[:, axis2], alpha=1,color='g',s=60)
  plt.scatter(autoencoder.transform(triang_func)[:, axis1], autoencoder.transform(triang_func)[:, axis2], alpha=1,color='m',s=60)
  plt.scatter(autoencoder.transform(teX)[:, axis1], autoencoder.transform(teX)[:, axis2], alpha=0.1,color='r')
  plt.xlabel('axis %d'%axis1)
  plt.ylabel('axis %d'%axis2)
  plt.show()
  fig, ax = plt.subplots()
  plt.plot(np.linspace(-5,25,npt),sine_func[0,:],'b')
  plt.plot(np.linspace(-5,25,npt),autoencoder.reconstruct(sine_func)[0,:],'r')
  plt.plot(np.linspace(-5,25,npt),square_func[0,:]+2,'g')
  plt.plot(np.linspace(-5,25,npt),autoencoder.reconstruct(square_func)[0,:]+2,'r')
  plt.plot(np.linspace(-5,25,npt),triang_func[0,:]+4,'m')
  plt.plot(np.linspace(-5,25,npt),autoencoder.reconstruct(triang_func)[0,:]+4,'r')
  plt.show()
  