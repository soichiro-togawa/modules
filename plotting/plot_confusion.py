def plot_confusion_matrix(y_train,y_train_pred_numpy,df_train,df_test):
  y_pred_classes_train = np.argmax(y_train_pred_numpy,axis = 1) 
  y_true = np.argmax(y_train,axis = 1) 
  confusion_mtx = confusion_matrix(y_true, y_pred_classes_train) 
  plt.figure(figsize=(7,7))
  targets = list(set(df_train.columns) ^ set(df_test.columns))
  target_classes = len(targets)
  # confusion_matrix with correlation map⇒外側の関数から内側の関数を読み込める
  def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
      """
      This function prints and plots the confusion matrix.
      Normalization can be applied by setting `normalize=True`.
      """
      import itertools
      plt.imshow(cm, interpolation='nearest', cmap=cmap)
      plt.title(title)
      plt.colorbar()
      tick_marks = np.arange(len(classes))
      plt.xticks(tick_marks, classes, rotation=45)
      plt.yticks(tick_marks, classes)

      if normalize:
          cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

      thresh = cm.max() / 2.
      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, cm[i, j],
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

      plt.tight_layout()
      plt.ylabel('True label')
      plt.xlabel('Predicted label')
  plot_confusion_matrix(confusion_mtx, classes = range(target_classes))

#実行
plot_confusion_matrix(y_val,y_val_pred,df_train,df_test)