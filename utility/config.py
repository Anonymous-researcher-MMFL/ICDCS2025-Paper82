def optimizer_config(name_data):
    if name_data =='cifar10':
        learning_rate = 0.1
        
    if name_data =='mnist':
        learning_rate = 0.01#0.001
        # OCt 4,previous 0.01
        
    if name_data=='fashion_mnist':
        learning_rate= 0.01 #0.002 # 2e-6
    if name_data=='fashion_mnist2':
        learning_rate= 0.01 #0.002 # 2e-6
    if name_data == 'emnist':
        learning_rate = 0.5 # 2e-6
        # in distribution experiment, need to use 0.5
        #
    if name_data == 'shakespeare':
        learning_rate = 1.4  # 0.002 # 2e-6
    return learning_rate