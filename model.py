import os
import sys
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

# ================= CONFIGURATION =================
MODEL_DIR = "model_v2"
MODEL_CHECKPOINT = os.path.join(MODEL_DIR, "lite_gpt_model.pt")
DATA_DIR = "data_v2"
CORPUS_FILE = os.path.join(DATA_DIR, "corpus.txt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-4
MAX_GENERATE_LEN = 50 
DEFAULT_TEMPERATURE = 1.0
MIN_CORPUS_LINES = 25000

# ================= LOGGING =================
logging.basicConfig(
    format="[LiteGPT] %(message)s",
    level=logging.INFO,
    stream=sys.stdout
)
chat_logger = logging.getLogger("lite_gpt_chat")

# ================= TOKENIZER =================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
special_tokens = {
    "pad_token": "<PAD>",
    "bos_token": "<BOS>",
    "eos_token": "<EOS>",
    "additional_special_tokens": ["<user>:", "<assistant>:"]
}
tokenizer.add_special_tokens(special_tokens)

# ================= SYNTHETIC CORPUS =================
def create_synthetic_corpus():
    chat_logger.info("Creating enriched synthetic conversational corpus...")

    user_starts = [
        "Hello!", "How are you?", "What's up?", "Tell me a joke!", "Do you know something interesting?",
        "Can you be sarcastic?", "Give me advice.", "What's your opinion on AI?", "Do you like music?",
        "How do I learn Python?", "What's the weather today?", "Do you know a fun fact?", "Can you explain quantum physics?",
        "Who invented the light bulb?", "What is the capital of France?", "How does a computer work?",
        "What's the meaning of life?", "Can you help me with a coding problem?", "Explain the solar system.",
        "What is machine learning?", "How is a large language model trained?", "What is the difference between a chatbot and a search engine?",
        "What's the best way to start a new project?", "Tell me a story.", "What are the benefits of exercise?",
        "How do I write a good prompt for an LLM?", "What is the history of AI?", "Can you give me a recipe for something?",
        "Why is the sky blue?", "What is the square root of 64?", "Tell me about the Roman Empire.",
        # New question about the model
        "What is your name?", "Who created you?", "Which model are you?",
        "What is language model?", "what is large language model?", "What is small language model?",
        "what is AI?", "what is artificial intelligence?", "what is machine learning?",
    ]

    assistant_replies = [
        "Hi there! 😎", "I'm doing great, thanks!", "Not much, you?",
        "Why did the chicken cross the road? To get to the other side!",
        "Absolutely! Here's a fun fact...", "Of course, but only if you want humorous. 😏",
        "I think learning by doing is the best way.", "AI is fascinating, isn't it?",
        "I love music! My favorite is electronic.", "Start with small projects and practice a lot!",
        "Looks sunny outside!", "Did you know octopuses have three hearts?", "Quantum physics is tricky, but fascinating!",
        "Thomas Edison is often credited, though many others contributed.", "The capital of France is Paris. It's a beautiful city!",
        "A computer works by processing data using a CPU and memory. It's a lot like a brain, but made of silicon.",
        "That's a classic! Many philosophers believe it's to find happiness and purpose.", "I can certainly try! What's the problem you're facing?",
        "The solar system consists of the sun and all the objects orbiting it, like planets and moons.",
        "Machine learning is a field of AI where computers learn from data without being explicitly programmed.",
        "It's like teaching a computer to read by showing it millions of books, and then asking it to write a new one.",
        "A chatbot is for conversation, while a search engine finds information. It's the difference between talking to a friend and searching a library.",
        "Start with a clear goal, break it down into small tasks, and don't be afraid to make mistakes!",
        "Once upon a time, in a digital land far, far away...", "Regular exercise can improve your physical and mental health. Just remember to stretch!",
        "A good prompt should be clear, specific, and provide context. Think of it like giving directions to a friend!",
        "AI has a long history, starting with early concepts in the 1950s and evolving rapidly with new technology.",
        "How about a simple recipe for chocolate chip cookies? They're always a hit!", "The sky is blue because of Rayleigh scattering, which makes blue light scatter more than other colors.",
        "The square root of 64 is 8. Easy peasy!", "The Roman Empire was a powerful civilization that controlled vast territories around the Mediterranean Sea for centuries.",
        # New answers about the model
        "I am LiteGPT, a small language model trained by Raziel1234 🙂",
        "My creator is Raziel1234, and I am called LiteGPT.",
        "I am LiteGPT, a text-based language model trained by Raziel1234.",
        "I'm a small language model trained by Raziel1234.",
        "language model is a type of AI that can understand and generate human Language.",
        "A Large language model is a type of AI that has been trained on a giant data and 100M-1B params. I will be happy to help you. 😊",
        "A small language model is a type of AI to generate text and 1M-10M params. Like me 😊. I will happy to help you. 😊🤣",
        "Sure! I’m happy to help you about AI😊! AI stands for artificial intelligence, AI can help you and understand like human 👨‍🦰.",
        "Artificial intelligence is machine learning, like human.", "AI is a machine like human. 😊!"
    ]

    conversations = []
    num_examples = MIN_CORPUS_LINES // 2

    for _ in range(num_examples):
        user_text = random.choice(user_starts)
        assistant_text = random.choice(assistant_replies)
        conversations.append(f"<BOS>\n<user>: {user_text}\n<assistant>: {assistant_text}\n<EOS>")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for line in conversations:
            f.write(line + "\n")

    chat_logger.info(f"Finished creating and saving {len(conversations)} conversations.")

# ================= LOAD & TOKENIZE =================
def load_corpus_and_tokenize():
    if not os.path.exists(CORPUS_FILE):
        create_synthetic_corpus()

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_chunks = []

    for line in lines:
        tokens = tokenizer.encode(line.strip())
        if len(tokens) < MAX_LENGTH:
            tokens += [tokenizer.pad_token_id] * (MAX_LENGTH - len(tokens))
        for i in range(0, len(tokens), MAX_LENGTH):
            chunk = tokens[i:i + MAX_LENGTH]
            if len(chunk) < MAX_LENGTH:
                chunk += [tokenizer.pad_token_id] * (MAX_LENGTH - len(chunk))
            all_chunks.append(chunk)

    data = torch.tensor(all_chunks, dtype=torch.long)
    return data

# ================= DATASET =================
class TokenDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return x, y

def collate_batch(batch):
    xb = torch.stack([item[0] for item in batch])
    yb = torch.stack([item[1] for item in batch])
    return xb.to(DEVICE), yb.to(DEVICE)

# ================= MODEL =================
class LiteGPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x

class LiteGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6, d_ff=1024, block_size=MAX_LENGTH, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([LiteGPTBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, x, targets=None):
        B, T = x.size()
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(positions)
        x = tok_emb + pos_emb

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None

# ================= DIALOGUE MANAGER =================
class DialogueManager:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def generate_response(self, user_text, max_new_tokens=MAX_GENERATE_LEN, temperature=DEFAULT_TEMPERATURE):
        self.model.eval()
        prompt = f"<BOS>\n<user>: {user_text}\n<assistant>:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids.tolist()[0]
        idx = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self.model(idx)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)
                if next_idx.item() == tokenizer.eos_token_id:
                    break
                idx = torch.cat([idx, next_idx], dim=1)
                generated.append(next_idx.item())

        # decode עם skip_special_tokens=True
        response = tokenizer.decode(generated, skip_special_tokens=True)
        # חיתוך אחרי <assistant>:
        if "<assistant>:" in response:
            response = response.split("<assistant>:")[-1]
        return response.strip()

# ================= MAIN =================
if __name__ == "__main__":
    data = load_corpus_and_tokenize()
    if len(data) == 0:
        raise ValueError("[LiteGPT] ERROR: DataLoader received empty dataset!")

    dataset = TokenDataset(data)
    model = LiteGPT(vocab_size=len(tokenizer)).to(DEVICE)

    if os.path.exists(MODEL_CHECKPOINT):
        try:
            model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
            chat_logger.info("Loaded existing model checkpoint.")
        except Exception as e:
            chat_logger.error(f"Error loading model: {e}. Retraining.")
            os.remove(MODEL_CHECKPOINT)

    if not os.path.exists(MODEL_CHECKPOINT):
        chat_logger.info("Training new model...")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            total_loss = 0
            model.train()
            for i, (xb, yb) in enumerate(dataloader):
                optimizer.zero_grad()
                logits, loss = model(xb, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if (i + 1) % 50 == 0:
                    chat_logger.info(f"Epoch {epoch+1}/{EPOCHS}, batch {i+1}/{len(dataloader)}, loss: {loss.item():.4f}")
            chat_logger.info(f"Epoch {epoch+1} complete. Avg loss: {total_loss / len(dataloader):.4f}")

        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_CHECKPOINT)
        chat_logger.info(f"Training complete. Model saved to {MODEL_CHECKPOINT}")

    dm = DialogueManager(model)
    chat_logger.info("LiteGPT Interactive Mode. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        response = dm.generate_response(user_input)
        print(f"LiteGPT: {response}")
