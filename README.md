# math_eq

\\recurrent neural network training 

title: online handwritten mathematics equation recognition (either browser based or tablet based education products).
summary: Imagine a kid works on math problems in her iPad and she writes down her answer using Apple Pencil. When iPad recognizes her answer is wrong, it provides relevant mini problems to help her to learn necessary concepts to solve the original problem. Meanwhile, her misconceptions are analyzed from her math equation answer and appropriate feedback is generated for her teachers or parents. This project is a baby step for this lofty goal. An essential technique of this learning platform is handwritten recognition system for math equations. As of today, a bidirectional RNN has been trained to recognize 53 math strokes (not symboles -- details comes below). Additional processing is planned to bring this recognition to symbol and equation level for complete math equation recognition.
There are all three github repositories --  1) data preparation (https://github.com/rosepark222/HW_Rcode), 2) model training (https://github.com/rosepark222/HW_symbol_learn), and 3) deployment (https://github.com/rosepark222/keras_deploy). 
I think the best way to have the feel for this project is to dabble with the deployment in Google cloud: http://pradoxum001.appspot.com/

detail: 1) designed bidirectional recurrent neural network 2) performed training, 3) examined hyperparameters 

A bidirectional RNN was trained. The dimensions of input , the first and second layers are 15, 64 and 32, respectively. LSTM cells were used to overcome vanishing gradient issue and better learning when there is need of forgetting prelearnined information. A typical case in natural language processing example, when a subject in the sentences has changed from singular and plural, the previously registered information of the subject being singular should be forgotten and replaced by the current information. In the current project, I think LSTM cells are useful when two strokes are very similar (from the pen-down and up to 80%, such as s and 8), then the last 20% are only unique information revealing the difference between two strokes. To make things worse, if one stroke can overlap with the other (such as ‘s’ and ‘8’), RNN cannot decide if the stroke is s or 8 during the first 70%-80% of stroke, because they are almost identical. How does RNN determine between two strokes using the last 20% of data available on only one of the stroke is not clear to me.  Empirical test showed that 8 is confused with s but s is not confused with 8. It indicates that the first 80% of trace+offline features are enough to determine if s is s but not enough to determine if 8 is 8 (I am not even sure this makes sense). An interesting note is that clockwisely written 8 was not confused with s, because the direction of the trace is totally different from that of s. There was similar confusion between left stroke of lower  case x (flipped c) and number 2. When I added offline features, this confusion *magically* disappeared. I believe this is because offline features were distinct between the two strokes. The good news is that this is not end of the road. I can always train Convolutional Neural Network (CNN) to distinguish strokes imposing challenge to RNN. Before we hastily conclude that RNN is not learning, I have to mention that the number of s and 8 are 170 and 760, respectively, that were duplicated (copied) to be 1000 to achieve the data balancing.  These data size seems way too small. If more data were ever added, RNN might as well begin to learn the difference between the two without any problem. The hope for the ML still lives on. 

The batch size was 500, the number of steps in epoch is 30 and the number of epochs for the conversion are slightly above 1,000. The learning rate was not adjusted at this moment. For this reason, there are some spikes once the loss goes down below 0.01. Even though I believe changing the learning rate would prevent the jittery behavior of optimization, it also depends on the quality of data as well. When I trained just handful of strokes (e.g., 5 confusing ones), the loss and mse metrics were much smaller than 0.01. I programmed Keras to save model parameters whenever the loss decreases during the training, so that it always remember the model resulted in the minimum loss during the training.

Prior to the training, data were balanced to have 1000 sample per strokes (data are copied when the numbers of a stroke is less than 1000). Total 53 strokes are trained. At the end of training, the loss, mse and accuracy were approximately in the order of 0.002, 3E-6, and .999. Splitting of training and test sets were useful at the beginning of the project. After the model was deployed to Google cloud, more online experiments were utilized to evaluate the performance. The interface to the deployment can be leveraged to collect handwritten strokes data. This may be a good future development, because currently data are lacking in some strokes. 

My personal impression of important factors that have improved the learning are adopting bidirectional RNN, adding offline features, balancing the data, removing mislabeled strokes. 
These factors are ordered in time that I have implemented, not in its magnitude of the effect. Other factors such as modifying the cell size in neural network or the number of layers have not investigated yet.
The learning is not perfect - meaning there are misclassifications. For example, 8 is recognized as s because it is similar in their features. As long as the confusions are consistent, however, the correct recognition can be accomplished using the context. Since the target application is question answering system in an e-learning platform, the set of symbols per a problem is significantly smaller than the entire set of symbols. For example, students are likely to use k in their answers if the stem of the problem uses k as a variable. The problem can even mandate students to use certain symbols to describe their answers. Based on this contextual information, developing probabilistic models for building symbols from strokes is the logical next step.  


conda install -c conda-forge keras tensorflow flask

or

conda env create -f conda.tf2.env.ymltxt 


> train[, .N, by=symbol_final][order(N)]
    symbol_final    N
 1:   \\lim_3_li   24
 2:  \\log_1_log   26
 3:     \\mu_1_1   28
 4:        l_1_1   38
 5:  \\sigma_1_1   42
 6:  \\cos_1_cos   48
 7:     \\gt_1_1   57
 8:  \\lim_2_lim   66
 9:  \\gamma_1_1   68
10:   \\log_2_lo   68
11:   \\cos_2_co   77
12:        o_1_1   85
13:        L_1_1   96
14:        w_1_1   99
15:   \\tan_3_an  100
16:      \\}_1_1  101
17:      \\{_1_1  101
18:        [_1_1  121
19:        q_1_1  143
20:        /_1_1  157
21:      p_2_ear  159
22:        s_1_1  172
23:        ]_1_1  194
24:        h_1_1  202
25:  \\sum_2_bot  212
26:        v_1_1  223
27:    \\sum_1_1  223
28:        5_1_1  236
29:  \\theta_1_1  238
30:        k_1_1  242
31:        u_1_1  244
32:        g_1_1  258
33:        p_1_1  269
34:        j_2_1  311
35:    f_2_cobra  317
36:     \\lt_1_1  328
37:   \\beta_1_1  354
38:        r_1_1  355
39:   y_2_flower  356
40:  \\infty_1_1  361
41:        m_1_1  443
42:        4_1_1  460
43:        e_1_1  526
44:     t_2_tail  541
45:  \\alpha_1_1  544
46:    \\int_1_1  594
47:        x_1_1  634
48:        i_2_1  653
49:     5_2_hook  681
50:        9_1_1  748
51:        8_1_1  758
52:        7_1_1  769
53:        6_1_1  811
54:        c_1_1  871
55:        d_1_1  925
56:        z_1_1 1345
57:     4_2_nose 1378
58:        b_1_1 1434
59:        y_1_1 1505
60:   \\sqrt_1_1 1738
61:        0_1_1 1746
62:        n_1_1 2061
63:        a_1_1 2212
64:        1_1_1 2406
65:        3_1_1 2815
66:    x_2_right 3030
67:     x_2_left 3210
68:        )_1_1 3671
69:        (_1_1 3696
70:        2_1_1 6774
    symbol_final    N
    
