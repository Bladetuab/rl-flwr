python run_campaign.py --rounds 30 --clients 50 --malicious 20 --local-epochs 10 \
      --datasets mnist,fashion_mnist \
      --attacks gaussian,lie,min_max,fang,no_attack \
      --defenses ustad_cal