## Before the improvement

network

```python
class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(
                16, 3
            ),  # output layer will be 3 , because damage_grade is a 3-class classification task
        )

    def forward(self, x):
        return self.layers(x)

```



outcome

```
Mean Accuracy: 0.5382
Accuracy on full training set: 0.5465
Classification Report:
               precision    recall  f1-score   support

           0       0.66      0.55      0.60       729
           1       0.53      0.91      0.67      1968
           2       0.00      0.00      0.00      1303

    accuracy                           0.55      4000
   macro avg       0.39      0.49      0.42      4000
weighted avg       0.38      0.55      0.44      4000
```



## After the 1st improvement

I add the

1. \# compute for weights for classes -- to deal with the problem of class's unbalance

   ```python
   class_weights = compute_class_weight("balanced", classes=np.unique(Y), y=Y)
   class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
   ```

   

2. \# Normalize for data -- (ljh key change point)

   ```python
   scaler = StandardScaler()
   
   X = scaler.fit_transform(X)
   
   X_test = scaler.transform(X_test)
   ```

   

3. Change the network ( it seems a little failed)

```python
class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),  # continue to lower dropout bility
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(
                16, 3
            ),  # output layer will be 3 , because damage_grade is a 3-class classification task
        )

    def forward(self, x):
        return self.layers(x)

```

outcome

```
Mean Accuracy: 0.5382
Accuracy on full training set: 0.5465

Mean Accuracy: 0.4682
Accuracy on full training set: 0.476

Classification Report:
               precision    recall  f1-score   support

           0       0.51      0.78      0.61       729
           1       0.65      0.09      0.16      1968
           2       0.44      0.88      0.59      1303

    accuracy                           0.48      4000
   macro avg       0.53      0.59      0.46      4000
weighted avg       0.55      0.48      0.39      4000
```

It's sad that the outcome become less, So I try to change the early stop longer first. (from delta 0.001 and patience 20 to (0.001,40), and also roll back to last net structure.

#### change back to before net and patience to 40

outcome

```
Mean Accuracy: 0.4813
Accuracy on full training set: 0.475

Classification Report:
               precision    recall  f1-score   support

           0       0.53      0.77      0.63       729
           1       0.61      0.10      0.17      1968
           2       0.44      0.88      0.58      1303

    accuracy                           0.47      4000
   macro avg       0.53      0.58      0.46      4000
weighted avg       0.54      0.47      0.39      4000
```

#### Cancel the normalization?

outcome

```
Mean Accuracy: 0.4788
Accuracy on full training set: 0.479

Classification Report:
               precision    recall  f1-score   support

           0       0.53      0.76      0.63       729
           1       0.59      0.12      0.20      1968
           2       0.44      0.87      0.58      1303

    accuracy                           0.48      4000
   macro avg       0.52      0.58      0.47      4000
weighted avg       0.53      0.48      0.40      4000
```

So I may need more experiment on that to get a better score.

