import numpy as np
def RetrievalEvaluation(C_depth, distM, model_label, depth_label):

	'''
	C_depth: retrieval number for the testing example, Nx1
	distM: distance matrix, row for testing example, column for training example
	model_label: model_label for training example
	depth_label: label for testing example
	'''

	nb_of_query = C_depth.shape[0]

	p_points = np.zeros((nb_of_query, np.amax(C_depth)))

	ap = np.zeros(nb_of_query)

	nn = np.zeros(nb_of_query)

	ft = np.zeros(nb_of_query)

	st = np.zeros(nb_of_query)

	dcg = np.zeros(nb_of_query)

	e_measure = np.zeros(nb_of_query)

	recall = np.zeros((distM.shape[0], distM.shape[1]))

	precision = np.zeros((distM.shape[0], distM.shape[1]))

        #print('The number of queries is {}'.format(nb_of_query))
	for qqq in range(nb_of_query):

		temp_dist = distM[qqq]
		#print 'The len of temp_list is %d' % ( len(temp_dist))
		s = list(temp_dist)
		R = sorted(range(len(s)), key=lambda k: s[k])
		#print type(R)
		#print R
		#print model_label.shape
		model_label_qqq = model_label[R]
		#print model_label_qqq.shape
		G = np.zeros(distM.shape[1])
		for i in range(distM.shape[1]):
			if model_label_qqq[i] == depth_label[qqq]:
				G[i] = 1
		G_sum = np.cumsum(G)
		#print G_sum
		#print G_sum.shape
		r1 = G_sum / C_depth[qqq]
		p1 = G_sum / np.arange(1, distM.shape[1]+1)

       		r_points = np.zeros(C_depth[qqq])
        	for i in range(C_depth[qqq]):
			temp = np.where(G_sum == i+1)
			#print type(temp)
			#print temp
	            	r_points[i] = np.where(G_sum == (i+1))[0][0] + 1
		#print p_points.shape
		#print G_sum.shape
		#print type(r_points)
		#print 'Here is r_points', r_points
		r_points_int = np.array(r_points, dtype=int)
       		p_points[qqq][:C_depth[qqq]] = G_sum[r_points_int-1] / r_points
       		ap[qqq] = np.mean(p_points[qqq][:C_depth[qqq]])

		#print 'Here is the average precision %f' % (ap[qqq])
       	 	nn[qqq] = G[0]

        	ft[qqq] = G_sum[C_depth[qqq]-1] / C_depth[qqq]
                #print('The shape of C_depths {}'.format(C_depth.shape))
	        st[qqq] = G_sum[min(2*C_depth[qqq]-1, C_depth.shape[0])] / C_depth[qqq]
       		p_32 = G_sum[31] / 32
	        r_32 = G_sum[31] /C_depth[qqq]

       		if p_32 == 0 and r_32 == 0:
           		e_measure[qqq] = 0
	        else:
       			e_measure[qqq] = 2* p_32 * r_32/(p_32+r_32)

		#print np.log2(np.arange(2,C_depth[qqq]+1))

	        NORM_VALUE = 1 + np.sum(1/np.log2(np.arange(2,C_depth[qqq]+1)))
       		dcg_i = 1/np.log2(np.arange(2, len(R)+1)) * G[1:]
        	#dcg_i = np.concatenate((np.array(G[0]), dcg_i), axis=0)
		dcg_i = np.insert(dcg_i, 0, G[0])
        	dcg[qqq] = np.sum(dcg_i, axis=0)/NORM_VALUE
        	recall[qqq] = r1
        	precision[qqq] = p1


	nn_av = np.mean(nn)
	ft_av = np.mean(ft)
	st_av = np.mean(st)
	dcg_av = np.mean(dcg)
	e_av = np.mean(e_measure)
	map_ = np.mean(ap)

	pre = np.mean(precision, axis=0)
	rec = np.mean(recall, axis=0)

	return nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec
