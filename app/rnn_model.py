import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size=10000,
        embedding_dim=100,
        hidden_dim=128
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden

    
class LSTMTextGenerator:
    def __init__(self, corpus, epochs=100):
        words = " ".join(corpus).lower().split()

        self.vocab = {"<UNK>": 0}
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

        self.inv_vocab = {
            index: word for word, index in self.vocab.items()
        }

        encoded = [self.vocab[word] for word in words]
        self.model = LSTMModel(vocab_size=len(self.vocab))

        inputs = torch.tensor(
            encoded[:-1], dtype=torch.long
        ).unsqueeze(0)
        targets = torch.tensor(
            encoded[1:], dtype=torch.long
        ).unsqueeze(0)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01
        )

        self.model.train()
        for _ in range(epochs):
            outputs, _ = self.model(inputs)
            loss = criterion(
                outputs.reshape(-1, len(self.vocab)),
                targets.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()

    def generate_text(self, start_word, length):
        words = start_word.lower().split()
        if not words:
            raise ValueError("start_word must not be empty")

        input_ids = [
            self.vocab.get(word, 0) for word in words
        ]
        input_tensor = torch.tensor(
            [input_ids], dtype=torch.long
        )
        hidden = None

        with torch.no_grad():
            for _ in range(length):
                outputs, hidden = self.model(
                    input_tensor, hidden
                )
                logits = outputs[0, -1].clone()
                logits[0] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()

                words.append(self.inv_vocab[next_id])
                input_tensor = torch.tensor(
                    [[next_id]], dtype=torch.long
                )

        return " ".join(words)