#-------------------------------------------------------------------
# Cluster and Cloud Computing Assignment 1
# Author Name: Rahul Sharma
# Student ID: 706453
# Email: sharma1@student.unimelb.edu.au
#
# Description: This code performs a word count on a large file (10GB)
# containing tweets from multiple users. The input for this code is
# the file_name and the search_word that the user wishes to locate.
# Following which, depending on the configurations specified in the PBS
# script, either the sequential/ serial version or the parallel version
# of the code will be executed. 
#
#
# This code can be invoked from the command prompt using the following
# command:  
#
#  mpirun -np 1 python mpigeneric.py <file_name> <search_word>
#
# This command would execute the sequential version, for parallel, depending
# on the number of nodes required, we can set it in the PBS script and in
# order to specify the cores we set the number 1 to 4 (for 4 cores) or 8 
# (for 8 cores) 
#
#
# This script is meant to break down the flow of logic and is not meant to
# show a proper functionally decomposed methodology to go about doing this.
#
#-------------------------------------------------------------------



from mpi4py import MPI
import sys, time, csv, re, operator, math

#-------------------------------------------------------------------
#
# MPI Initial set-up
#

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()
read = MPI.MODE_RDONLY
term_occurrences = 0
total_count = []
total_tweeters = []
total_topics = []
tweeter_dict = {}
topic_dict = {}
chunk_size = 8


#-------------------------------------------------------------------
#
# Arguments taken from the user during execution. The arguments accepted
# are the file_name and the search_word to be looked for. If the user 
# does not provide any arguments then the default settings will be used 
# to perdorm the word count
#

if len(sys.argv) > 1:
	file_name = sys.argv[1]
else:
	file_name = 'twitter.csv'
if len(sys.argv) > 2:
	search_term = sys.argv[2]
else:
	search_term = 'love'


#-------------------------------------------------------------------
#
# Start the time in order to measure the time taken for execution of 
# the word count.
#

if rank == 0:
	start_time = time.time()


#-------------------------------------------------------------------
#
# Reading the file. The reading operation is performed using MPI-IO.
# The concept was researched from here: http://goo.gl/Yiu5fq
#

read_file = MPI.File.Open(comm, file_name, read)
length_of_file = MPI.File.Get_size(read_file)


#-------------------------------------------------------------------
#
# First the sequential version of the code is defined. If the size < 2
# then the code will be run sequentially or serially. The first code
# explored by me was the sequential version in order for me to use it
# as benchmark for verifying the counts and the sequence of the top
# tweeters and topics present in the large twitter.csv file.
#

if size < 2:

	# The Data is divided into chunks in order to process it in parts
	# rather then all at once, thus avoiding the step memory exceeded
	# error often seen in Spartan.
	buffer_length = int(math.ceil(length_of_file/chunk_size)) 

	print("The configuration being run is the serial version where size = 1")
	print(" ")

	# Have data in terms of Bytes
	buff = bytearray(buffer_length)

	# Loop through each chunk (currently 8) to get the count and aggregate
	# it using regex.
	for i in range(chunk_size):

		# Create the offset to specify to the core the size of the buffer
		# to read and process
		offset = buffer_length * i
		read_file.Read_at_all(offset, buff)

		# Only for testing (not required for sequential runs)
		#comm.Barrier()

		# Find the number of times the search_word appears in the tweet file
		regex = b'\\b' + bytes(search_term,'utf-8') + b'\\b'
		m = re.findall(regex,buff,re.I)

		# Only for testing (not required for sequential runs)
		#comm.Barrier()
		total_count.append(m)

		# Find all the tweeters mentioned in the tweet file
		m = re.findall(b'@\w+',buff,re.I)

		# Since the data is in the form of Byte Array, we first need to decode
		# it otherwise we will get an error: Unhashable type - Byte Array
		flattened_tweeters = [str(a.decode('utf-8')).lower() for a in m]
		for twitter_user in flattened_tweeters:
			if twitter_user in tweeter_dict:
				tweeter_dict[twitter_user] += 1
			else:
				tweeter_dict[twitter_user] = 1
 

		# Only for testing (not required for sequential runs)
		#comm.Barrier()

		# Find all the topics mentioned in the tweet file
		m = re.findall(b'#\w+',buff,re.I)

		# Since all the data is in the form of Byte Array, we first need to decode
		# it otherwise we will get an error: Unhashable type - Byte Array
		flattened_topics = [str(a.decode('utf-8')).lower() for a in m]
		for user_topic in flattened_topics:
			if user_topic in topic_dict:
				topic_dict[user_topic] += 1
			else:
				topic_dict[user_topic] = 1
		
		# Only for testing (not required for sequential runs)
		#comm.Barrier()
	
	# Close the file after all the chunks are iterated over.	
	read_file.Close()
else:

	# This states the buffer length depending on the number of cores
	# available.
	buffer_length = int(math.ceil(length_of_file/size)) 
	
	# Have data in terms of Bytes
	buff = bytearray(buffer_length)

	# Each rank is given its offset value to read and process from.
	offset = buffer_length * rank
	read_file.Read_at_all(offset, buff)
	
	comm.Barrier()
	
	# Once the file is read, close it. 
	read_file.Close()
	
	# Find the number of times the search_word appears in the tweet file
	regex = b'\\b' + bytes(search_term,'utf-8') + b'\\b'
	m = re.findall(regex,buff,re.I)

	comm.Barrier()

	# Gather the lists for the total count of the search_word from all the ranks
	total_count = comm.gather(m,root=0)

	# Find all the tweeters mentioned in the tweet file
	m = re.findall(b'@\w+',buff)

	comm.Barrier()

	# Gather the lists for all the tweeters found from all the ranks
	total_tweeters = comm.gather(m,root=0)

	# Find all the topics mentioned in the tweet file
	m = re.findall(b'#\w+',buff)

	comm.Barrier()

	# Gather the lists for all the topics found from all the ranks
	total_topics = comm.gather(m,root=0)



#-------------------------------------------------------------------
#
# If we are at the Master (Rank = 0) then start aggregating the results 
# count for the search_word and print it out.
# 
# Starting from here, all parts are to be run on the Master. These parts
# have currently been segregated into smaller blocks with specific comments
# to describe what those blocks do or are used for. Each block has a
# corresponding 'if' statement only to state that purpose, and should be
# removed and merged to create one set of statements to be run under one
# if statement.
#

if rank == 0:
	for i in total_count:
		term_occurrences += len(i)

	if size > 2:
		print("This is the parallel version where size = %s" % size)
		print(" ")

	print("The total count for the search term '"+ str(search_term) +"' is %d" % (term_occurrences))
	

#-------------------------------------------------------------------
#
# If we are at the Master (Rank = 0) then flatten the lists of lists 
# gathered from all the cores and decode them (as they are in Byte Array), 
# following which include them in a dictionary if that tweeter was not 
# present previously and set its value to 1, else increase the count for 
# that tweeter by 1.
#

if rank == 0:
	if size > 2:
		flattened_tweeters = [str(a.decode('utf-8')).lower() for b in total_tweeters for a in b]
		for twitter_user in flattened_tweeters:
			if twitter_user in tweeter_dict:
				tweeter_dict[twitter_user] += 1
			else:
				tweeter_dict[twitter_user] = 1

		

#-------------------------------------------------------------------
#
# If we are at the Master (Rank = 0) then flatten the list of lists 
# gathered from all the cores and decode them (as they are in Byte Array), 
# following which include them in a dictionary if that topic was not 
# present previously and set its value to 1, else  increase the count 
# for that topic by 1
#

if rank == 0:
	if size > 2:
		flattened_topics = [str(a.decode('utf-8')).lower() for b in total_topics for a in b]
		for user_topic in flattened_topics:
			if user_topic in topic_dict:
				topic_dict[user_topic] += 1
			else:
				topic_dict[user_topic] = 1

		
#-------------------------------------------------------------------
# Finally, sort the dictionary created in descending order and limit 
# the list by 10 using list splicing to display the top 10 tweeters 
# and top 10 topics present in the large tweet file given to us.
#

if rank == 0:
	sort_tweeter = sorted(tweeter_dict.items(),key=operator.itemgetter(1),reverse = True)
	sort_topic = sorted(topic_dict.items(),key=operator.itemgetter(1),reverse=True)
	
	print(" ")
	print("Top 10 tweeters are:")
	print(" ")
	
	for index, (key,value) in enumerate(sort_tweeter[0:10]):
		print(str(index+1) + ' ' + str(key) + ' - ' + str(value))

	print(" ")
	print("Top 10 topics are:")
	print(" ")

	for index, (key,value) in enumerate(sort_topic[0:10]):
		print(str(index+1) + ' ' + str(key) + ' - ' + str(value))
	#print("Top 10 tweeters are: ", sort_tweeter[0:10])
	#print("Top 20 topics are: ", sort_topic[0:10])
	end_time = time.time()
	print(" ")
	print("Total time taken to run this process is " +  str(end_time - start_time) + " sec")

