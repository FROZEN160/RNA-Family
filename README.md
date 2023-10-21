# ConF: A deep learning model for Noncoding RNA Families Prediction
<div style="text-align:center">
    <img src="https://github.com/FROZEN160/RNA-Family/assets/80303403/a9fbab9b-2362-44ef-af68-67ae7b61a99f" alt="image" width="600" height="700" />
</div>

This work introduces a deep learning model called ConF, designed for accurate and efficient prediction of non-coding RNA (NcRNA) families. NcRNAs are essential RNA molecules involved in various cellular processes, including replication, transcription, and gene expression. Predicting NcRNA families is crucial for in-depth RNA research, as NcRNAs within the same family often exhibit similar functions.

Traditional experimental methods for identifying NcRNA families are often time-consuming and labor-intensive, while computational approaches relying on annotated secondary structure data face limitations in handling complex structures like pseudoknots, resulting in suboptimal prediction performance. To overcome these challenges, the ConF model integrates a range of advanced techniques, including residual networks, dilated convolutions, and cross multi-head attention mechanisms. By employing a combination of dual-layer convolutional networks and bidirectional long short-term memory networks (BiLSTM), ConF effectively captures intricate features embedded within RNA sequences. This feature extraction process significantly improves prediction accuracy compared to existing methods.

Experimental evaluations conducted on a widely used dataset demonstrate the outstanding performance of the ConF model, including accuracy, sensitivity, and other performance metrics. Overall, the ConF model represents a promising solution for accurate and efficient prediction of NcRNA families, overcoming the limitations of traditional experimental and computational methods. The code for this work is publicly available.

## Environment Setup  
tensorflow-gpu               2.9.1

pandas                       1.4.3

keras                        2.9.0

Keras-Preprocessing          1.1.2

sklearn                      0.0

numpy                        1.23.0

matplotlib                   3.5.2

## User Guide
We have already uploaded the dataset and code used to this link. You just need to change the data reading address in the code to apply it.

## Personal Email Address
If you have any questions regarding the paper, you can contact us by sending an email to "frozen@mail.dlut.edu.cn".
