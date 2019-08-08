args=split(getArgument(), ",")
infilepath=args[0]
outfilepath=args[1]
open("infilepath");
run("Movie...", "frame=100 container=.mp4 using=H.264 video=normal save=outfilepath");
close();
