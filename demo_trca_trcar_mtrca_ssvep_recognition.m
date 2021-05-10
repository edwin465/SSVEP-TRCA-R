%% Demo of TRCA and its variants for SSVEP Recognition %%%
% This codes shows how to use the task-related component analysis
% (TRCA), the TRCA with reference signal (TRCA-R), and the modified 
% TRCA (mTRCA) for SSVEP recognition.

% Please refer the following papers for more details:
% Wong, C. M., et al. (2020). Spatial Filtering in SSVEP-based BCIs: Unified Framework and New Improvements. IEEE Transactions on Biomedical Engineering,

% This code is prepared by Chi Man Wong (chiman465@gmail.com)
% Date: 
% 27 July 2020 (v1.0): TRCA and TRCA-R
% 29 July 2020 (v1.1): Add mTRCA (Tang, J. et al. Optimizing SSVEP-Based 
% BCI System towards Practical High-Speed Spelling. Sensors 2020, 20, 4186.)

% if you use this code for a publication, please cite the following paper

% @article{wong2020spatial,
%   title={Spatial Filtering in SSVEP-based BCIs: Unified Framework and New Improvements},
%   author={Wong, Chi Man and Wang, Boyu and Wang, Ze and Lao, Ka Fai and Rosa, Agostinho and Wan, Feng},
%   journal={IEEE Transactions on Biomedical Engineering},
%   year={2020},
%   publisher={IEEE}
% }

clear all;
close all;
is_demo=1;
% Please download the SSVEP benchmark dataset for this code
% Wang, Y., et al. (2016). A benchmark dataset for SSVEP-based brain-computer interfaces. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(10), 1746-1752.
% Then indicate where the directory of the dataset is :
if is_demo==1
    str_dir=cd;
    num_of_subj=1; % Number of subjects 
    ch_used=[1:9];
else
    str_dir='D:\matlab_work\Tsinghua dataset 2016'; % Directory of the SSVEP Dataset (Change it if necessary)
    num_of_subj=35; % Number of subjects (35 if you have the benchmark dataset)
    ch_used=[48 54 55 56 57 58 61 62 63]; % Pz, PO5, PO3, POz, PO4, PO6, O1,Oz, O2 (in SSVEP benchmark dataset)
end



Fs=250; % sample rate

num_of_trials=2;                    % Number of training trials (1<=num_of_trials<=2)
num_of_harmonics=5;                 % Number of harmonics (for reference signal)
TRCAwithRef=1;                      % For TRCA-R (1: with Ref., 0: without Ref.)

% time-window length (min_length:delta_t:max_length)
min_length=0.5;
delta_t=0.1;
max_length=0.5;                     % [min_length:delta_t:max_length]

enable_bit=[1 1 1 1];               % Select the algorithms: bit 1: CCA, bit 2: TRCA, bit 3: TRCA-R, bit 4: mTRCA, e.g., enable_bit=[1 1 1 1]; -> select all algorithms
is_center_std=1;                    % 0: without , 1: with (zero mean, and unity standard deviation)

% Chebyshev Type I filter design
[b2,a2] = cheby1(4,1,[7/(Fs/2) 90/(Fs/2)],'bandpass');


seed = RandStream('mt19937ar','Seed','shuffle');
for sn=1:num_of_subj
    tic
    if is_demo==1
        load(strcat(str_dir,'\','exampleData.mat'));
    else
        load(strcat(str_dir,'\S',num2str(sn),'.mat'));
    end
    
    %  pre-stimulus period: 0.5 sec
    %  latency period: 0.14 sec
    eeg=data(ch_used,floor(0.5*Fs+0.14*Fs):floor(0.5*Fs+0.14*Fs)+4*Fs-1,:,:);
    
    
    [d1_,d2_,d3_,d4_]=size(eeg);
    d1=d3_;d2=d4_;d3=d1_;d4=d2_;
    no_of_class=d1;
    % d1: num of stimuli
    % d2: num of trials
    % d3: num of channels % Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
    % d4: num of sampling points
    for i=1:1:d1
        for j=1:1:d2
            y=reshape(eeg(:,:,i,j),d3,d4);
            SSVEPdata(:,:,j,i)=reshape(y,d3,d4,1,1);
            
            for ch_no=1:d3
                y_sb(ch_no,:)=filtfilt(b2,a2,y(ch_no,:));
            end
            
            SSVEPdata(:,:,j,i)=reshape(y_sb,d3,d4,1,1);
            
        end
    end
    
    clear eeg
    %% Initialization
    
    n_ch=size(SSVEPdata,1);
    
    TW=min_length:delta_t:max_length;
    TW_p=round(TW*Fs);
    n_run=d2;                                % number of used runs
    
    pha_val=[0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 ...
        0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5]*pi;
    sti_f=[8:0.2:15.8];
    n_sti=length(sti_f);                     % number of stimulus frequencies
    temp=reshape([1:40],8,5);
    temp=temp';
    target_order=temp(:)';
    SSVEPdata=SSVEPdata(:,:,:,target_order);
    
    n_correct=zeros(length(TW),length(enable_bit)); % Count how many correct detections
    n_detection=zeros(length(TW));
    
    seq_0=zeros(d2,num_of_trials);
    for run=1:d2
        %         % leave-one-run-out cross-validation
        
        if (num_of_trials==1)
            seq1=run;
        elseif (num_of_trials==d2-1)
            seq1=[1:n_run];
            seq1(run)=[];
        else
            % leave-one-run-out cross-validation
            % Randomly select the trials for training
            isOK=0;
            while (isOK==0)
                seq=randperm(seed,d2);
                seq1=seq(1:num_of_trials);
                seq1=sort(seq1);
                if isempty(find(sum((seq1'*ones(1,d2)-seq_0').^2)==0))
                    isOK=1;
                end
            end
            
        end
        idx_traindata=seq1; % index of the training trials
        idx_testdata=1:n_run; % index of the testing trials
        idx_testdata(seq1)=[];
        
        for i=1:no_of_class
            if length(idx_traindata)>1
                signal_template(i,:,:)=mean(SSVEPdata(:,:,idx_traindata,i),3);
            else
                signal_template(i,:,:)=SSVEPdata(:,:,idx_traindata,i);
            end
        end
        
        
        for run_test=1:length(idx_testdata)
            for tw_length=1:length(TW)
                sig_len=TW_p(tw_length);
                test_signal=zeros(d3,sig_len);
                fprintf('Testing TW %fs, No.crossvalidation %d \n',TW(tw_length),idx_testdata(run_test));
                
                for i=1:no_of_class
                    %                     for sub_band=1:num_of_subbands
                    test_signal=SSVEPdata(:,1:TW_p(tw_length),idx_testdata(run_test),i);
                    if (is_center_std==1)
                        test_signal=test_signal-mean(test_signal,2)*ones(1,length(test_signal));
                        test_signal=test_signal./(std(test_signal')'*ones(1,length(test_signal)));
                    end
                    for j=1:no_of_class
                        template=reshape(signal_template(j,:,[1:sig_len]),d3,sig_len);
                        if (is_center_std==1)
                            template=template-mean(template,2)*ones(1,length(template));
                            template=template./(std(template')'*ones(1,length(template)));
                        end
                        
                        % Generate the sine-cosine reference signal
                        ref1=ref_signal_nh(sti_f(j),Fs,pha_val(j),sig_len,num_of_harmonics);
                        % ================ CCA ===============
                        if (enable_bit(1)==1)
                            [~,~,r0]=canoncorr(test_signal',ref1');
                            CCAR(j)=r0(1);
                        else
                            
                            CCAR(j)=0;
                        end
                        %===============TRCA==================
                        if (enable_bit(2)==1)
                            if (num_of_trials==1)
                                % num_of_trials cannot be less than 2
                                % in TRCA
                                TRCAR(j)=0;
                            else
                                if ((i==1) && (j==1))
                                    W_eTRCA=[];
                                    for jj=1:no_of_class
                                        trca_X2=[];
                                        trca_X1=zeros(d3,sig_len);
                                        for tr=1:num_of_trials
                                            X0=reshape(SSVEPdata(:,1:sig_len,idx_traindata(tr),jj),d3,sig_len);
                                            if (is_center_std==1)
                                                X0=X0-mean(X0,2)*ones(1,length(X0));
                                                X0=X0./(std(X0')'*ones(1,length(X0)));
                                            end
                                            trca_X1=trca_X1+X0;
                                            trca_X2=[trca_X2;X0'];
                                        end
                                        S=trca_X1*trca_X1'-trca_X2'*trca_X2;
                                        Q=trca_X2'*trca_X2;
                                        [eig_v1,eig_d1]=eig(Q\S);
                                        [eig_val,sort_idx]=sort(diag(eig_d1),'descend');
                                        eig_vec=eig_v1(:,sort_idx);
                                        W_eTRCA=[W_eTRCA; eig_vec(:,1)'];
                                    end
                                end
                                
                                cr1=corrcoef(W_eTRCA*test_signal,W_eTRCA*template);
                                TRCAR(j)=cr1(1,2);
                            end
                        else
                            TRCAR(j)=0;
                        end
                        %===============TRCA-R==================
                        if (enable_bit(3)==1)
                            if (num_of_trials==1)
                                % num_of_trials cannot be less than 2
                                % in eTRCA
                                TRCARR(j)=0;
                            else
                                if ((i==1) && (j==1))
                                    W_TRCAR=[];
                                    for jj=1:no_of_class
                                        trca_X=[];
                                        if (TRCAwithRef==1)
                                            Ref=ref_signal_nh(sti_f(jj),Fs,pha_val(jj),sig_len,num_of_harmonics);
                                            [Q_ref1,R_ref1]=qr(Ref',0);
                                            ref_matrix=Q_ref1*Q_ref1';
                                        else
                                            ref_matrix=eye(sig_len);
                                        end
                                        %
                                        
                                        LL=repmat(ref_matrix,num_of_trials);
                                        
                                        if (num_of_trials==5)
                                            LL=LL-blkdiag(ref_matrix,ref_matrix,ref_matrix,ref_matrix,ref_matrix);
                                        elseif (num_of_trials==4)
                                            LL=LL-blkdiag(ref_matrix,ref_matrix,ref_matrix,ref_matrix);
                                        elseif (num_of_trials==3)
                                            LL=LL-blkdiag(ref_matrix,ref_matrix,ref_matrix);
                                        elseif (num_of_trials==2)
                                            LL=LL-blkdiag(ref_matrix,ref_matrix);
                                        else
                                        end
                                        
                                        for tr=1:num_of_trials
                                            X0=reshape(SSVEPdata(:,1:sig_len,idx_traindata(tr),jj),d3,sig_len);
                                            if (is_center_std==1)
                                                X0=X0-mean(X0,2)*ones(1,length(X0));
                                                X0=X0./(std(X0')'*ones(1,length(X0)));
                                            end
                                            trca_X=[trca_X;X0'];
                                        end
                                        S=trca_X'*LL*trca_X;
                                        Q=trca_X'*trca_X;
                                        [eig_v1,eig_d1]=eig(Q\S);
                                        [eig_val,sort_idx]=sort(diag(eig_d1),'descend');
                                        eig_vec=eig_v1(:,sort_idx);
                                        W_TRCAR=[W_TRCAR; eig_vec(:,1)'];
                                    end
                                end
                                cr1=corrcoef(W_TRCAR*test_signal,W_TRCAR*template);
                                TRCARR(j)=cr1(1,2);
                            end
                        else
                            TRCARR(j)=0;
                        end
                        %===============modified TRCA==================
                        if (enable_bit(4)==1)
                            if (num_of_trials==1)
                                % num_of_trials cannot be less than 2
                                % in TRCA
                                mTRCAR(j)=0;
                            else
                                if ((i==1) && (j==1))
                                    W_mTRCA=[];
                                    trca_X1=zeros(d3,sig_len,no_of_class);
                                    trca_X2=zeros(d3,sig_len*num_of_trials,no_of_class);
                                    for kk=1:no_of_class
                                        for tr=1:num_of_trials
                                            X0=reshape(SSVEPdata(:,1:sig_len,idx_traindata(tr),kk),d3,sig_len);
                                            if (is_center_std==1)
                                                X0=X0-mean(X0,2)*ones(1,length(X0));
                                                X0=X0./(std(X0')'*ones(1,length(X0)));
                                            end
                                            trca_X1(:,:,kk)=trca_X1(:,:,kk)+X0;
                                            trca_X2(:,(tr-1)*sig_len+1:tr*sig_len,kk)=X0;
                                        end
                                    end
                                    for jj=1:no_of_class    
                                        S1=trca_X1(:,:,jj)*trca_X1(:,:,jj)'-trca_X2(:,:,jj)*trca_X2(:,:,jj)'; 
                                        Q1=trca_X2(:,:,jj)*trca_X2(:,:,jj)';    
                                        
                                        Q0=zeros(size(Q));
                                        S0=zeros(size(S));
                                        for kk=1:no_of_class
                                            if kk==jj
                                            else
                                                S0=S0+trca_X1(:,:,jj)*trca_X1(:,:,kk)'+trca_X1(:,:,kk)*trca_X1(:,:,jj)';                                                    
                                                Q0=Q0+trca_X2(:,:,kk)*trca_X2(:,:,kk)';
                                            end
                                        end
                                        S=2*S1-S0/(no_of_class-1);
                                        Q=(Q1+Q0/(no_of_class-1))/2;
                                        
                                        [eig_v1,eig_d1]=eig(Q\S);
                                        [eig_val,sort_idx]=sort(diag(eig_d1),'descend');
                                        eig_vec=eig_v1(:,sort_idx);
                                        W_mTRCA=[W_mTRCA; eig_vec(:,1)'];
                                    end
                                end
                                
                                cr1=corrcoef(W_mTRCA*test_signal,W_mTRCA*template);
                                MTRCAR(j)=cr1(1,2);
                            end
                        else
                            MTRCAR(j)=0;
                        end
                    end
                    
                    
                    
                    CCAR1=CCAR;
                    TRCAR1=TRCAR;
                    TRCARR1=TRCARR;
                    MTRCAR1=MTRCAR;
                    
                    
                    [~,idx]=max(CCAR1);
                    if idx==i
                        n_correct(tw_length,1)=n_correct(tw_length,1)+1;
                    end
                    [~,idx]=max(TRCAR1);
                    if idx==i
                        n_correct(tw_length,2)=n_correct(tw_length,2)+1;
                    end
                    [~,idx]=max(TRCARR1);
                    if idx==i
                        n_correct(tw_length,3)=n_correct(tw_length,3)+1;
                    end
                    [~,idx]=max(MTRCAR1);
                    if idx==i
                        n_correct(tw_length,4)=n_correct(tw_length,4)+1;
                    end
                    n_detection(tw_length)=n_detection(tw_length)+1;
                end
            end
        end
        idx_train_run(run,:)=idx_traindata;
        idx_test_run(run,:)=idx_testdata;
        seq_0(run,:)=seq1;
    end
    
    
    %% Save results
    toc
    accuracy=100*n_correct./n_detection
    % column 1: CCA
    % column 2: TRCA
    % column 3: TRCA-R
    xlswrite('acc_file.xlsx',accuracy'/100,strcat('Sheet',num2str(sn)));
    disp(sn)
end


