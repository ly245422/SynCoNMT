import re
import os
import io
import time
import jieba
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import legacy as legacy_optimizers

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def preprocess_eng(text):
    text = text.lower().strip()

    # 单词和标点之间加空格
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    text = re.sub(r"([?.!,])", r" \1 ",text)
    # 多个空格合并为一个
    text = re.sub(r'[" "]+', " ", text)

    # 除了(a-z, A-Z, ".", "?", "!", ",")这些字符外，全替换成空格
    text = re.sub(r"[^a-zA-Z?.!,]+", " ", text)
    text = text.rstrip().strip()

    # 增加开始结束标志，让模型知道何时停止预测
    text = '<start> ' + text + ' <end>'
    return text

def preprocess_chn(text):
    text = text.lower().strip()
    text = jieba.cut(text, cut_all=False, HMM=True)
    text = " ".join(list(text))  # 词之间增加空格
    text = '<start> ' + text + ' <end>'
    return text


en_sentence = "May I borrow this book?"
chn_sentence = "我可以借这本书吗？"
# print(preprocess_eng(en_sentence))
# print(preprocess_chn(chn_sentence))

path_to_file = "data/cmn.txt" 
en_texts, cn_texts = [], []
for line in open(path_to_file, encoding='UTF-8').read().strip().split('\n'):
    en_text, cn_text = line.split('\t')
    en_text_prep, cn_text_prep = preprocess_eng(en_text), preprocess_chn(cn_text)
    en_texts.append(en_text_prep)
    cn_texts.append(cn_text_prep)


en_converter = tf.keras.preprocessing.text.Tokenizer(filters='')
en_converter.fit_on_texts(en_texts)
cn_converter = tf.keras.preprocessing.text.Tokenizer(filters='')
cn_converter.fit_on_texts(cn_texts)

en_text_ids = en_converter.texts_to_sequences(en_texts)
cn_text_ids = cn_converter.texts_to_sequences(cn_texts)

# print(en_texts[0])
# print(en_text_ids[0])

# print(cn_texts[0])
# print(cn_text_ids[0])

max_en_len = max([len(seq) for seq in en_text_ids])
max_cn_len = max([len(seq) for seq in cn_text_ids])
# print("英文最大长度：{}".format(max_en_len))
# print("中文最大长度：{}".format(max_cn_len))

en_text_ids_padded = tf.keras.preprocessing.sequence.pad_sequences(en_text_ids, maxlen=max_en_len, padding='post')
cn_text_ids_padded = tf.keras.preprocessing.sequence.pad_sequences(cn_text_ids, maxlen=max_cn_len, padding='post')

# print(en_texts[0])
# print(en_text_ids_padded[0])

# print(cn_texts[0])
# print(cn_text_ids_padded[0])

# 中文为源语言，英文为目标语言
# 分割训练数据和验证数据
input_cn_train, input_cn_val, target_en_train, target_en_val = train_test_split(
    cn_text_ids_padded, en_text_ids_padded, test_size=0.05)

# 显示训练数据和验证数据的大小
# print(len(input_cn_train), len(target_en_train),
#       len(input_cn_val), len(target_en_val))

# 先做shuffle， 再取batch
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((input_cn_train, target_en_train))
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)

valid_dataset = tf.data.Dataset.from_tensor_slices((input_cn_val, target_en_val))
valid_dataset = valid_dataset.batch(batch_size=batch_size, drop_remainder=True)

# next(iter(valid_dataset))

# 模型训练
embedding_dim = 256
hidden_units = 1024
# 0 是为padding保留的一个特殊id， 所以要 + 1
cn_vocab_size = len(cn_converter.word_index) + 1
en_vocab_size = len(en_converter.word_index) + 1

class ComplexEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(ComplexEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = tf.keras.layers.Embedding(input_dim, output_dim)

    def call(self, inputs):
        real_part = self.embedding(inputs)
        imag_part = tf.zeros_like(real_part)
        return tf.complex(real_part, imag_part)


class ComplexGRU(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ComplexGRU, self).__init__(**kwargs)
        self.units = units
        self.gru_real = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.gru_imag = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, inputs, initial_state):
        real_inputs = tf.math.real(inputs)
        imag_inputs = tf.math.imag(inputs)

        real_output, real_state = self.gru_real(real_inputs, initial_state=tf.math.real(initial_state))
        imag_output, imag_state = self.gru_imag(imag_inputs, initial_state=tf.math.imag(initial_state))

        output = tf.complex(real_output, imag_output)
        state = tf.complex(real_state, imag_state)

        return output, state

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = ComplexEmbedding(vocab_size, embedding_dim)
        self.gru = ComplexGRU(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.complex(tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units)))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # Convert complex inputs to real parts for Dense layers
        query_real = tf.math.real(query)
        values_real = tf.math.real(values)

        # Expand dimensions for time axis calculation
        hidden_with_time_axis = tf.expand_dims(query_real, 1)

        # Calculate attention scores
        score = self.V(tf.nn.tanh(self.W1(values_real) + self.W2(hidden_with_time_axis)))

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # Calculate context vector
        context_vector_real = tf.reduce_sum(attention_weights * values_real, axis=1)

        # Convert context vector back to complex
        context_vector = tf.complex(context_vector_real, tf.zeros_like(context_vector_real))

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = ComplexEmbedding(vocab_size, embedding_dim)
        self.gru = ComplexGRU(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)

        # attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x, hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, state, attention_weights


encoder = Encoder(cn_vocab_size, embedding_dim, hidden_units, batch_size)
decoder = Decoder(en_vocab_size, embedding_dim, hidden_units, batch_size)

# 定义优化器和损失
optimizer = legacy_optimizers.Adam(learning_rate=0.0005)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    """Calculate the loss value

    Args:
        real: the true label  shape == (batch_size,) -> (64,)
        pred: the probability of each word from the vocabulary, is the output from the decoder 
                 shape == (batch_size, vocab_size) -> (64, 6082)

    Returns: 
        the average loss of the data in a batch size
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# 定义 checkpoint
checkpoint_dir = './checkpoint'
checkpoint_prefix = checkpoint_dir +  '/seq2seq_mt_ckpt'
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
fine_tune = True
pretrain_ckpt = "pre_model/pretrain_ckpt"
if fine_tune:
    checkpoint.restore(pretrain_ckpt)

# 定义训练函数

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        # decoder的第一个输入为 '<start>'
        # dec input shape == (batch_size, 1) -> (64, 1)
        dec_input = tf.expand_dims(
            [en_converter.word_index['<start>']] * batch_size, 1)
        # 将当前输出作为下一步的输入
        # 第一个输入为 <start>, 所有 t 从 1 开始 (不是 0)
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(
                dec_input, dec_hidden, enc_output)

            # 计算当前timestep的损失
            loss += loss_function(targ[:, t], predictions)

            # 更新输入为上一时刻的输出
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    # 聚合所有参数
    variables = encoder.trainable_variables + decoder.trainable_variables

    # 计算梯度
    gradients = tape.gradient(loss, variables)

    # 根据梯度更新变量
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

# 启动训练
EPOCHS = 100  # 50 测试需要，设置训练轮数为3， 实际为保证效果，建议设置为50

# 记载训练过程
# 设置文件路径
txt_file_path = "epoch_loss.txt"
if os.path.exists(txt_file_path):
    os.remove(txt_file_path)
    
# 创建并写入内容到txt文件
def write_to_txt(file_path, content):
    with open(file_path, 'a') as file:
        file.write(content)
    

for epoch in range(EPOCHS):
    start = time.time()

    # 获取gru的初始状态
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(iter(train_dataset)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))

    # 每两个迭代保存一次模型
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    epoch_info = 'Epoch {} Loss {:.4f}\nTime taken for 1 epoch {} sec\n'.format(epoch + 1, total_loss, time.time() - start)
    write_to_txt(txt_file_path, epoch_info)
    
## Define inference function to handle complex predictions
def inference(sentence):
    attention_plot = np.zeros((max_en_len, max_cn_len), dtype=np.float32)

    sentence = preprocess_chn(sentence)

    # Convert words to IDs
    inputs = [cn_converter.word_index.get(word, 0) for word in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_cn_len, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    # Initialize hidden state for encoder
    hidden = tf.complex(tf.zeros((1, hidden_units)), tf.zeros((1, hidden_units)))

    # Encode the input sentence
    enc_out, enc_hidden = encoder(inputs, hidden)

    # Initialize decoder input with '<start>'
    dec_input = tf.expand_dims([en_converter.word_index['<start>']], 0)

    # Initialize decoder hidden state with encoder final hidden state
    dec_hidden = enc_hidden

    for t in range(max_en_len):
        # Generate predictions and attention weights
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # Store attention weights for visualization
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        # Get predicted word ID
        predicted_id = tf.argmax(predictions[0]).numpy()

        # Get predicted word from index
        predicted_word = en_converter.index_word.get(predicted_id, '<end>')

        # Append predicted word to result
        result += predicted_word + ' '

        # Stop predicting if '<end>' is predicted
        if predicted_word == '<end>':
            return result, sentence, attention_plot

        # Prepare next decoder input as the current predicted ID
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

## Define function to plot attention weights
def plot_attention(attention, sentence, predicted_sentence, save_path=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    # Specify font path
    font_path = "/root/autodl-tmp/SimSun.ttf"
    if os.path.exists(font_path):
        mpl.font_manager.fontManager.addfont(font_path)
    
    # Set font properties
    fontdict = {'fontsize': 14, 'family': 'SimSun'}
    
    # Set x and y axis ticks and labels
    ax.set_xticks(np.arange(len(sentence)))
    ax.set_yticks(np.arange(len(predicted_sentence)))
    ax.set_xticklabels(sentence, fontproperties=mpl.font_manager.FontProperties(fname=font_path), fontdict=fontdict)
    ax.set_yticklabels(predicted_sentence, fontproperties=mpl.font_manager.FontProperties(fname=font_path), fontdict=fontdict)
    
    # Set major locators for x and y axes
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    # Save plot if specified
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

## Define function to translate a sentence
def translate(sentence):
    result, sentence, attention_plot = inference(sentence)

    print('输入: %s' % (sentence))
    print('翻译结果: {}'.format(result))
    
    # Specify path to save attention plot
    save_path = "/root/autodl-tmp/result.png"
    
    # Crop attention plot to the length of the translated sentence
    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    
    # Plot and save attention weights visualization
    plot_attention(attention_plot, sentence.split(' '), result.split(' '), save_path)

## Restore the latest checkpoint and perform translation
checkpoint_dir = '/root/autodl-tmp/checkpoint'
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print("Successfully restored checkpoint:", latest_checkpoint)
    translate('他不适合当老师。')
else:
    print("No checkpoint found in:", checkpoint_dir)
