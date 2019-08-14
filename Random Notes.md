# Random Notes

Ethan Human Accuracy
- Ravdess acc: 67.01%
- Tess acc: 89.23%

2^7 = 128
1/128 = 0.781%

- Remember the last batch has inputs of zero and labels of all zero
- Remove first and last second of Ravdess because it's quiet
- We use a majority voting method to address disagreement among voters
- E.g. [N, A, N, A, H] -> [N, A, H] -> [2, 2, 1] -> [N, A]
- Exact algorithm
  - If only only emotion present, one-hot encode it
  - Else k-hot encode the label
    - Count each emotion that's present
    - Divide the label by the highest count
    - Floor the label to remove any emotions that weren't voted the highest
    - If the label is all zeros, don't use the same
    - Else return the label
- Issues with max voting
  - Loss of num total votes (confidence in vote)
  - Loss of smaller votes (ignore emotions that are close/ambiguous)
- What's the line between a sample that has two emotions conveyed verses ambiguous emotions?
  - 2 labels: {composition or ambiguity}
  - 3 labels: {ambiguity}
- 5 label sample: 1040_TAI_ANG_XX with [A, S, D, N, F]
- Another limitation is how we mapped calm to neutral and pleasant surprise to surprise.
- Compound emotions
  - Excited: {happy, surprise}
  - Curious: {fear, surprise}
- Map
  - Frustration -> anger
  - Excitment - > happy
- 