%% Demo of TRCA and TRCA-R for SSVEP Recognition %%%
% In this code, we provide how to use the task-related component analysis 
% (TRCA) and task-related component analysis with reference signal
% (TRCA-R) for SSVEP recognition.

% Please refer the following papers for more details:
% Wong, C. M., et al. (2019). Spatial Filtering in SSVEP-based BCIs: Unified Framework and New Improvements. IEEE Transactions on Biomedical Engineering,

% This code is prepared by Chi Man Wong (chiman465@gmail.com)
% Date: 27 July 2020
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
% Please download the SSVEP benchmark dataset for this code
% Wang, Y., et al. (2016). A benchmark dataset for SSVEP-based brain-computer interfaces. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(10), 1746-1752.
% Then indicate where the directory of the dataset is :
str_dir=cd; % Directory of the SSVEP Dataset (Change it if necessary)

num_of_subj=1; % Number of subjects (35 if you have the benchmark dataset)

Fs=250; % sample rate
% ch_used=[48 54 55 56 57 58 61 62 63]; % Pz, PO5, PO3, POz, PO4, PO6, O1,Oz, O2 (in SSVEP benchmark dataset)
ch_used=[1:9];

num_of_trials=2;                    % Number of training trials (1<=num_of_trials<=2)
num_of_harmonics=5;                 % for all cca-based methods
% num_of_subbands=5;                  % for filter bank analysis
% FB_coef0=[1:num_of_subbands].^(-1.25)+0.25; % for filter bank analysis
% About the above parameter, please check the related paper:
% Chen, X., et al. (2015). Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain¡Vcomputer interface. Journal of neural engineering, 12(4), 046008.

% time-window length (min_length:delta_t:max_length)
min_length=0.5;
delta_t=0.1;
max_length=0.5;                     % [min_length:delta_t:max_length]

enable_bit=[1 1 1];                 % Select the algorithms: bit 1: CCA, bit 2: TRCA, bit 3: TRCA-R, e.g., enable_bit=[1 1 1]; -> select all algorithms
is_center_std=1;                    % 0: without , 1: with (zero mean, and unity standard deviation)

% Chebyshev Type I filter design
[b2,a2] = cheby1(4,1,[7/(Fs/2) 90/(Fs/2)],'bandpass');
% for k=1:num_of_subbands
%     bandpass1(1)=8*k;
%     bandpass1(2)=90;
%     [b2(k,:),a2(k,:)] = cheby1(4,1,[bandpass1(1)/(Fs/2) bandpass1(2)/(Fs/2)],'bandpass');
% end

seed = RandStream('mt19937ar','Seed','shuffle');
for sn=1:num_of_subj
    tic
    load(strcat(str_dir,'\','exampleData.mat'));
%     load(strcat(str_dir,'\s',num2str(sn),'.mat'));
    
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
            
%             for sub_band=1:num_of_subbands
                
                for ch_no=1:d3
%                     if (num_of_subbands==1)
%                         y_sb(ch_no,:)=y(ch_no,:);
%                     else
                        y_sb(ch_no,:)=filtfilt(b2,a2,y(ch_no,:));
%                     end
                end
                
                SSVEPdata(:,:,j,i)=reshape(y_sb,d3,d4,1,1);
%             end
            
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
%     for sub_band=1:num_of_subbands
%         subband_signal(sub_band).SSVEPdata=subband_signal(sub_band).SSVEPdata(:,:,:,target_order); % To sort the orders of the data as 8.0, 8.2, 8.4, ..., 15.8 Hz
%     end
    
    
%     FB_coef=FB_coef0'*ones(1,n_sti);
    n_correct=zeros(length(TW),3); % Count how many correct detections
    
    
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
%             for k=1:num_of_subbands
%                 if length(idx_traindata)>1
%                     subband_signal(k).signal_template(i,:,:)=mean(subband_signal(k).SSVEPdata(:,:,idx_traindata,i),3);
%                 else
%                     subband_signal(k).signal_template(i,:,:)=subband_signal(k).SSVEPdata(:,:,idx_traindata,i);
%                 end
%             end
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
                            % ================ eCCA ===============
                            if (enable_bit(1)==1)
                                [~,~,r0]=canoncorr(test_signal',ref1');
                                CCAR(j)=r0(1);
%                                 [ecca_r1,CR(sub_band,j),itR(sub_band,j),CCAR(sub_band,j)]=extendedCCA(test_signal,ref1,template,num_of_r);
                            else
%                                 CR(sub_band,j)=0;
%                                 itR(sub_band,j)=0;
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
                                            Ref=ref_signal_nh(sti_f(jj),Fs,pha_val(jj),sig_len,num_of_harmonics);                                                                                        
                                            [Q_ref1,R_ref1]=qr(Ref',0);
                                            ref_matrix=Q_ref1*Q_ref1';

%                                             ref_matrix=eye(sig_len); 
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
                            
                        end
                        
%                     end
                    
                    CCAR1=CCAR;                    
                    TRCAR1=TRCAR;
                    TRCARR1=TRCARR;
                    
                    
                    [~,idx]=max(CCAR1);
                    if idx==i
                        n_correct(tw_length,1)=n_correct(tw_length,1)+1;
                    end                    
                    [~,idx]=max(TRCAR1);
                    if idx==i
                        n_correct(tw_length,2)=n_correct(tw_length,2)+1;
                    end
                    [~,idx]=max(TRCARR);
                    if idx==i
                        n_correct(tw_length,3)=n_correct(tw_length,3)+1;
                    end
                end
            end
        end
        idx_train_run(run,:)=idx_traindata;
        idx_test_run(run,:)=idx_testdata;
        seq_0(run,:)=seq1;
    end
    
    
    %% Save results
    toc
    accuracy=100*n_correct/n_sti/n_run/length(idx_testdata)
    % column 1: CCA
    % column 2: TRCA
    % column 3: TRCA-R    
    xlswrite('acc_file.xlsx',accuracy'/100,strcat('Sheet',num2str(sn)));
    disp(sn)
end


