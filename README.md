# March Madness Networks 2017

## Model Type
The model this builds is a mirror image with 2 inputs into each side. There is an LSTM (with shared-weights) on each side that sees a sample of games for teams A and B. Then there is a season-level vector input for each.

After the model processes the LSTM and the flat inputs, it predicts if team A will beat team B.
