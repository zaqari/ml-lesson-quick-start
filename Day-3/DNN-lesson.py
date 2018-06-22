import tempfile
import tensorflow as tf
import pandas as pd
import numpy as np

###LOGGING SET-UP###
tf.logging.set_verbosity(tf.logging.INFO)

Train_Data = "~/data/train-data.csv"
Test_Data = "~/data/test-data.csv"


df_train1=pd.read_csv(Train_Data, skipinitialspace=True)
df_test1=pd.read_csv(Test_Data, skipinitialspace=True)

feature_columns = ['ID', 'Q5A', 'Q5B', 'Q5C', 'Q6A', 'Q6B', 'Q6C','Q7A', 'Q7B', 'Q7C']

df_train = df_train1.replace([np.nan], '0')
#pd.concat([df_train1[feature_columns].astype(str), df_train1['Q4'].astype(int)], axis=1, join='inner')
df_test = df_test1.replace([np.nan], '0')
#pd.concat([df_test1[feature_columns].astype(str), df_test1['Q4'].astype(int)], axis=1, join='inner')

nD=4
CATEGORICAL_COLUMNS = ['Q5B', 'Q5C',
                       #'Q6B', 'Q6C',
                       #'Q7B', 'Q7C'
                       ]
LABELS_COLUMN = ['Q5A']
nClasses=5

def input_fn(df):
        # Creates a dictionary mapping from each continuous feature column name (k) to
        # the values of that column stored in a constant Tensor.
        #continuous_cols = {k: tf.constant(df[k].values)
                                #  for k in CONTINUOUS_COLUMNS}
        # Creates a dictionary mapping from each categorical feature column name (k)
        # to the values of that column stored in a tf.SparseTensor.
        categorical_cols = {k: tf.SparseTensor(
                indices=[[i, 0] for i in range(df[k].size)],
                values=df[k].values,
                dense_shape=[df[k].size, 1])
                        for k in CATEGORICAL_COLUMNS}
        # Merges the two dictionaries into one.
        feature_cols = dict(categorical_cols.items())
        # Converts the label column into a constant Tensor.
        label = tf.constant(df[LABELS_COLUMN].values.astype(int))
        # Returns the feature columns and the label.
        return feature_cols, label

def train_input_fn():
        return input_fn(df_train)

def eval_input_fn():
        return input_fn(df_test)


ID=tf.contrib.layers.sparse_column_with_hash_bucket("ID", hash_bucket_size=int(1000))
#Q5A=tf.contrib.layers.real_valued_column("Q5A", dtype=tf.float32)
Q5B=tf.contrib.layers.sparse_column_with_hash_bucket("Q5B", hash_bucket_size=int(1000))
Q5C=tf.contrib.layers.sparse_column_with_hash_bucket("Q5C", hash_bucket_size=int(1000))
#Q6A=tf.contrib.layers.real_valued_column("Q6A", dtype=tf.float32)
Q6B=tf.contrib.layers.sparse_column_with_hash_bucket("Q6B", hash_bucket_size=int(1000))
Q6C=tf.contrib.layers.sparse_column_with_hash_bucket("Q6C", hash_bucket_size=int(1000))
#Q7A=tf.contrib.layers.real_valued_column("Q7A", dtype=tf.float32)
Q7B=tf.contrib.layers.sparse_column_with_hash_bucket("Q7B", hash_bucket_size=int(1000))
Q7C=tf.contrib.layers.sparse_column_with_hash_bucket("Q7C", hash_bucket_size=int(1000))

#INTERACTIONS
Q5 = tf.contrib.layers.crossed_column(
	[Q5B, Q5C],
	hash_bucket_size=int(1e6),
	combiner='sum')

Q6 = tf.contrib.layers.crossed_column(
	[Q6B, Q6C],
	hash_bucket_size=int(1e6),
	combiner='sum')

Q7 = tf.contrib.layers.crossed_column(
	[Q7B, Q7C],
	hash_bucket_size=int(1e6),
	combiner='sum')


deep_columns = [
        tf.contrib.layers.embedding_column(Q5, dimension=nD),
        tf.contrib.layers.embedding_column(Q6, dimension=nD),
        tf.contrib.layers.embedding_column(Q7, dimension=nD),

        tf.contrib.layers.embedding_column(Q5B, dimension=nD),
        tf.contrib.layers.embedding_column(Q5C, dimension=nD),
        tf.contrib.layers.embedding_column(Q6B, dimension=nD),
        tf.contrib.layers.embedding_column(Q6C, dimension=nD),
        tf.contrib.layers.embedding_column(Q7B, dimension=nD),
        tf.contrib.layers.embedding_column(Q7C, dimension=nD),
        ]

wide_columns =[]

pred_m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir='/tmp/DNN-model',
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50],
        n_classes=nClasses,
        #config=tf.contrib.learn.RunConfig(save_checkpoints_secs=10),
        fix_global_step_increment_bug=True
        )

m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir='/tmp/DNN-model',
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50],
        n_classes=nClasses,
        dnn_dropout=.5,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=40),
        fix_global_step_increment_bug=True)

input('Continue?')

m.fit(input_fn=train_input_fn, steps=300)
results = pred_m.evaluate(input_fn=eval_input_fn, steps=len(df_test))
print(results)
