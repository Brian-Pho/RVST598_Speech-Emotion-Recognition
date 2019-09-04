Recognizing multiple emotions in speech using machine learning

---

The goal of my research was to explore whether computers can detect multiple emotions in speech. This research was experimental as I had to write the computer program to detect the emotions and I had to test it. 

Current state-of-the-art systems can detect which emotions (such as happy, sad, anger) are present in speech with about a 50% accuracy but they are limited to one emotion per speech sample (such as a sentence). For example, one of these systems might output that a sentence was said in a surprised tone, but it could not say if it was happy and surprise (like receiving a gift) or if it was sad and surprise (like receiving news of the death of a loved one).

My research improves on the state-of-the-art by considering the multiple emotions case. I built a system using machine learning to automatically output whether multiple emotions are present in a speech sample. The system might output that this speech sample was “neutral” and “sad” or that this other sample is “disgust”. The way the system learns to classify speech samples is by feeding thousands of speech samples with their corresponding emotion into a machine learning model. The model makes a guess of what emotion is present and if the guess does not match the label, then we update the model using this feedback. Over time, this feedback helps the model learn which parts of speech are useful in determining which emotion is present. The system I created achieved an accuracy around 50% which I consider a success.

---

The role I played in this research included designing, building, and testing the system. While I did not come up with the research idea, I was the one that executed the idea and did most of the work. I collected the speech samples and wrote the software to detect the emotions. I did have two other students help me a few times but the majority of the work was done by me. I also wrote the paper and presentation that came out of this research that I have presented to my lab, the Biometrics Technologies Lab. 
