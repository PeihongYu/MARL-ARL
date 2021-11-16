## Single Agent Training
## Green
#python train.py --env MiniGrid-MultiAgent-N2-S4-A1G-v0 --algo ppo --obs partial --share-reward False --frames 200000
#python train.py --env MiniGrid-MultiAgent-N2-S4-A1G-v0 --algo ppo --obs full --share-reward False --frames 1000000
## Red
#python train.py --env MiniGrid-MultiAgent-N2-S4-A1R-v0 --algo ppo --obs partial --share-reward False --frames 200000
#python train.py --env MiniGrid-MultiAgent-N2-S4-A1R-v0 --algo ppo --obs full --share-reward False --frames 1000000
#
## Multi Agent Training
#python train.py --env MiniGrid-MultiAgent-N2-S4-A2-v0 --algo ppo --obs partial --share-reward False --frames 1600000
#python train.py --env MiniGrid-MultiAgent-N2-S4-A2-v0 --algo ppo --obs partial --share-reward True --frames 1600000
python train.py --env MiniGrid-MultiAgent-N2-S4-A2-v0 --algo ppo --obs full --share-reward True --batch-size 2048 --frames 100000000
python train.py --env MiniGrid-MultiAgent-N2-S4-A2-v0 --algo ppo --obs full --share-reward False --batch-size 2048 --frames 100000000