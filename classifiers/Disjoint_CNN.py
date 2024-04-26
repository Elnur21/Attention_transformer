import time
from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling1D, Input,SeparableConv1D,DepthwiseConv1D, Flatten,Dense,Reshape, BatchNormalization, ELU, Permute, MaxPooling1D
from classifiers.classifiers import predict_model
from utils.classifier_tools import create_class_weight
from utils.tools import save_logs
from classifiers.SelfAttention import self_attention


class Classifier_Disjoint_CNN:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        if verbose:
            print('Creating Disjoint_CNN Classifier')
        self.verbose = verbose
        self.output_directory = output_directory
        # Build Model -----------------------------------------------------------
        self.model = self.build_model(input_shape, nb_classes)
        # -----------------------------------------------------------------------
        if verbose:
            self.model.summary()
        # self.model.save_weights(self.output_directory + 'model_init.weights.h5')

    def build_model(self, input_shape, nb_classes):
        print("input_shape")
        print(input_shape)
        head,maxlen,_ = input_shape
        return self_attention(head, nb_classes,maxlen)

    def fit(self, Ximg_train, yimg_train, Ximg_val, yimg_val, epochs, batch_size):
        if self.verbose:
            print('[Disjoint_CNN] Training Custom_Disjoint_CNN Classifier')

        mini_batch_size = int(min(Ximg_train.shape[0] / 10, batch_size))
        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)
        file_path = self.output_directory + 'best_model.keras'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
        self.callbacks = [reduce_lr, model_checkpoint]

        # create class weights based on the y label proportions for each class
        print(yimg_train.shape)
        class_weight = create_class_weight(yimg_train)
        start_time = time.time()
        self.hist = self.model.fit(Ximg_train, yimg_train,
                                   validation_data=[Ximg_val, yimg_val],
                                   class_weight=class_weight,
                                   verbose=self.verbose,
                                   epochs=epochs,
                                   batch_size=mini_batch_size,
                                   callbacks=self.callbacks)
        self.duration = time.time() - start_time

        keras.models.save_model(self.model, self.output_directory + 'model.keras')
        print('[Disjoint_CNN] Training done!, took {}s'.format(self.duration))

    def predict(self, X_img, y_img, best):
        # if best:
        print(self.output_directory)
        model = keras.models.load_model(self.output_directory + 'best_model.keras')
        # else:
        #     model = keras.models.load_model(self.output_directory + 'model.keras')
        model_metrics, conf_mat, y_true, y_pred = predict_model(model, X_img, y_img, self.output_directory)
        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)
        keras.backend.clear_session()
        return model_metrics, conf_mat
