import matplotlib.pyplot as plt
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--what',
                        choices=['embed', 'embedg', 'hiddeng',
                                 'thresh', 'layer'])
    args = parser.parse_args()

    defaults = {'embed': 300,
                'embedg': 300,
                'hiddeng': 50,
                'thresh': 2,
                'layer': 3}

    if args.what == 'embedg':
        embed_sizes = [50, 100, 200, 300]
        for e in embed_sizes:
            file_name = os.path.join('logs',
                                     'MCG_inc_{}_50_3_2_g'.format(e),
                                     'output')
            losses = []
            with open(file_name, 'r') as f:
                for line in f:
                    if line.startswith('Step'):
                        losses.append(float(line.split()[-1]))

            plt.plot(range(len(losses)), losses, label=e)
            plt.xlabel('Optimization steps.', fontsize=20)
            plt.xticks(fontsize=20)
            plt.ylabel('Loss', fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.legend()
            #break
            plt.savefig('images/embed_glove_{}.pdf'.format(e), format='pdf')
            plt.clf()
            # plt.show()
    elif args.what == 'hiddeng':
        hidden_sizes = [50, 300, 500]
        for h in hidden_sizes:
            file_name = os.path.join('logs',
                                     'MCG_inc_300_{}_3_2_g'.format(h),
                                     'output')
            losses = []
            with open(file_name, 'r') as f:
                for line in f:
                    if line.startswith('Step'):
                        losses.append(float(line.split()[-1]))

            print(losses)
            plt.plot(range(len(losses)), losses, label=h)
            plt.xlabel('Optimization steps.', fontsize=20)
            plt.xticks(fontsize=20)
            plt.ylabel('Loss', fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.legend()
            # break
            plt.savefig('images/hidden_glove_{}.pdf'.format(h), format='pdf')
            plt.clf()
            # plt.show()

    elif args.what == 'thresh':
        thresh = [2, 3, 4]
        for t in thresh:
            file_name = os.path.join('logs',
                                     'MCG_inc_300_50_3_{}_g'.format(t),
                                     'output')
            losses = []
            with open(file_name, 'r') as f:
                for line in f:
                    if line.startswith('Step'):
                        losses.append(float(line.split()[-1]))

            print(losses)
            plt.plot(range(len(losses)), losses, label=t)
            plt.xlabel('Optimization steps.', fontsize=20)
            plt.xticks(fontsize=20)
            plt.ylabel('Loss', fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.legend()
            # break
            plt.savefig('images/thresh_{}.pdf'.format(t), format='pdf')
            plt.clf()
            # plt.show()

    elif args.what == 'layer':
        layers = [2, 3, 4]
        for l in layers:
            file_name = os.path.join('logs',
                                     'MCG_inc_300_50_{}_2_g'.format(l),
                                     'output')
            losses = []
            with open(file_name, 'r') as f:
                for line in f:
                    if line.startswith('Step'):
                        losses.append(float(line.split()[-1]))

            print(losses)
            plt.plot(range(len(losses)), losses, label=l)
            plt.xlabel('Optimization steps.', fontsize=20)
            plt.xticks(fontsize=20)
            plt.ylabel('Loss', fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.legend()
            # break
            plt.savefig('images/layers_{}.pdf'.format(l), format='pdf')
            plt.clf()
            # plt.show()
