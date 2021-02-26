import pstats
p = pstats.Stats('./server/profile')
p.strip_dirs()
p.sort_stats('cumtime')
p.print_stats(50)
