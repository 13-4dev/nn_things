import torch
import matplotlib.pyplot as plt  
from model import Transformer
from transformers import AutoTokenizer
from utils import (
    BATCH_SIZE,
    BLOCK_SIZE,
    DEVICE,
    DROPOUT,
    LEARNING_RATE,
    NUM_EMBED,
    NUM_HEAD,
    NUM_LAYER,
    MAX_ITER,
    EVAL_INTER,
    encode,
    decode,
    get_batch,
    save_model_to_chekpoint,
    estimate_loss,
)

path_do_data = r"data\english.txt"
data_raw = open(path_do_data, encoding="utf-8").read()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

data = encode(text_seq=data_raw, tokenizer=tokenizer)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

model = Transformer(
    vocab_size=vocab_size,
    num_embed=NUM_EMBED,
    block_size=BLOCK_SIZE,
    num_heads=NUM_HEAD,
    num_layers=NUM_LAYER,
    dropout=DROPOUT,
)
m = model.to(DEVICE)
print("Model with {:.2f}M parameters".format(sum(p.numel() for p in m.parameters()) / 1e6))
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

train_losses = []
val_losses = []

plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Train Loss')
line2, = ax.plot([], [], label='Validation Loss')
plt.xlabel('Evaluation Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

def update_plot():
    line1.set_xdata(range(len(train_losses)))
    line1.set_ydata(train_losses)
    line2.set_xdata(range(len(val_losses)))
    line2.set_ydata(val_losses)
    ax.relim() 
    ax.autoscale_view() 
    fig.canvas.draw() 
    plt.pause(0.001)  

for step in range(MAX_ITER):

    if step % EVAL_INTER == 0 or step == MAX_ITER - 1:
        loss_train = estimate_loss(
            data=train_data, model=m, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
        )
        loss_val = estimate_loss(
            data=val_data, model=m, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
        )
        train_losses.append(loss_train)
        val_losses.append(loss_val)
        print("step {:10} | train loss {:6.4f} | val loss {:6.4f}".format(step, loss_train, loss_val))
        
        update_plot()

    xb, yb = get_batch(data=train_data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
    logits, loss = m.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

save_model_to_chekpoint(model=m, path_to_checkpoint="checkpoints", epoch=step)

plt.ioff()
plt.show()

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(
    decode(
        enc_sec=m.generate(idx=context, max_new_tokens=100, block_size=BLOCK_SIZE)[0],
        tokenizer=tokenizer,
    )
)
