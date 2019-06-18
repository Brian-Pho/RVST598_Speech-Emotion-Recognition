# The list of seven emotions considered. Based on Ekman, 1999.
NEU = "NEUTRAL"
ANG = "ANGER"
DIS = "DISGUST"
FEA = "FEAR"
HAP = "HAPPY"
SAD = "SAD"
SUR = "SURPRISE"

# The standard index for each emotion. This is how we keep the encoding of
# emotion consistent across databases. We do this because it'll help the
# neural network learn which tensor maps to which emotion. This will be useful
# in Keras's "to_categorical()" function.
# E.g. Anger maps to 1 and surprise maps to 6.
EMOTION_MAP = {
    NEU: 0,
    ANG: 1,
    DIS: 2,
    FEA: 3,
    HAP: 4,
    SAD: 5,
    SUR: 6,
}