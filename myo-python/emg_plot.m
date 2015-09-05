rest = load('emg_data_rest.csv');
finger3 = load('emg_data_finger3.csv');

for i = 6:6
%     figure;
%     ax(1) = subplot(2,8,i);
    ax(1) = plot(rest(:,i));
%     xlim(ax(1),[0,500]);
%     hist(rest(:,i),25);
%     ax(1,i) = gca;
%     figure;
%     ax(2) = subplot(2,8,8+i);
figure ;
    ax(2) = plot(finger3(:,i),'r');
%     xlim(ax(2),[0,500]);
%     hist(finger3(:,i),25);
%     ax(2,i) = gca;
%     linkaxes(ax(:,i),'xy');

end