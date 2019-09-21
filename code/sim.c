#include "crun.h"

#ifndef TIME
#define TIME 0
#endif


/* Compute ideal load factor (ILF) for node */
static inline double neighbor_ilf(state_t *s, int nid) {
    graph_t *g = s->g;
    int outdegree = g->neighbor_start[nid+1] - g->neighbor_start[nid] - 1;
    int *start = &g->neighbor[g->neighbor_start[nid]+1];
    int i;
    double sum = 0.0;
    int lcount = s->rat_count[nid];
    for (i = 0; i < outdegree; i++) {
	int rcount = s->rat_count[start[i]];
	double r = imbalance(lcount, rcount);
	sum += r;
    }
    double ilf = BASE_ILF + 0.5 * (sum/outdegree);
    return ilf;
}

/* Compute weight for node nid */
static inline double compute_weight(state_t *s, int nid) {
    int count = s->rat_count[nid];
    double ilf = neighbor_ilf(s, nid);
    return mweight((double) count/s->load_factor, ilf);
}

/* Compute sum of weights in region of nid */
static inline double compute_sum_weight(state_t *s, int nid) {
    graph_t *g = s->g;
    double sum = 0.0;
    int eid;
    int eid_start = g->neighbor_start[nid];
    int eid_end  = g->neighbor_start[nid+1];
    int *neighbor = g->neighbor;
    for (eid = eid_start; eid < eid_end; eid++) 
	{
		int nbrnid = neighbor[eid];
		double w = compute_weight(s, nbrnid);
		s->node_weight[nbrnid] = w;
		sum += w;
    }
    return sum;
}

#if DEBUG
/** USEFUL DEBUGGING CODE **/
static void show_weights(state_t *s) {
    int nid, eid;
    graph_t *g = s->g;
    int nnode = g->nnode;
    int *neighbor = g->neighbor;
    outmsg("Weights\n");
    for (nid = 0; nid < nnode; nid++) {
	int eid_start = g->neighbor_start[nid];
	int eid_end  = g->neighbor_start[nid+1];
	outmsg("%d: [sum = %.3f]", nid, compute_sum_weight(s, nid));
	for (eid = eid_start; eid < eid_end; eid++) {
	    outmsg(" %.3f", compute_weight(s, neighbor[eid]));
	}
	outmsg("\n");
    }
}
#endif

/* Recompute all node counts according to rat population */
static inline void take_census(state_t *s) 
{
    graph_t *g = s->g;
    int nnode = g->nnode;
    int *rat_position = s->rat_position;
    int *rat_count = s->rat_count;
    int nrat = s->nrat;

    memset(rat_count, 0, nnode * sizeof(int));
    int ri;
    for (ri = 0; ri < nrat; ri++) {
	rat_count[rat_position[ri]]++;
    }
}

/* Recompute all node weights */
static inline void compute_all_weights(state_t *s) {
    int nid, i;
    graph_t *g = s->g;
    double *node_weight = s->node_weight;
    int nHubs = g->hub_count;

    #pragma omp for nowait
    for(i=0; i<nHubs; i++){
      int hub_id = g->hubs[i];
      node_weight[hub_id] = compute_weight(s, hub_id);
    }
    
    #pragma omp for
    for (nid = 0; nid < g->nnode; nid++){
      int eidStart = g->neighbor_start[nid];
      int eidEnd = g->neighbor_start[nid+1];
      if( eidEnd - eidStart <= 6 ){
        node_weight[nid] = compute_weight(s, nid);
      }
    }
}

/* In synchronous or batch mode, can precompute sums for each region */
static inline void find_all_sums(state_t *s) {
    graph_t *g = s->g;
    int nHubs = g->hub_count;
    int nNodes = g->nnode;
	    int nid, eid, i;
    
    #pragma omp single
    init_sum_weight(s);

    #pragma omp for nowait
    for(i=0; i<nHubs; i++){
      int hub_id = g->hubs[i];
      double sum = 0.0;
      for(eid = g->neighbor_start[hub_id]; eid < g->neighbor_start[hub_id+1]; eid++){
        sum += s->node_weight[g->neighbor[eid]];
        s->neighbor_accum_weight[eid] = sum;
      }
      s->sum_weight[hub_id] = sum;
    }
    
    #pragma omp for
    for (nid = 0; nid < nNodes; nid++) {
	double sum = 0.0;
        int eidStart = g->neighbor_start[nid];
        int eidEnd = g->neighbor_start[nid+1];
        if( eidEnd - eidStart <= 6 ){
          for (eid = eidStart; eid < eidEnd; eid++) {
            sum += s->node_weight[g->neighbor[eid]];
            s->neighbor_accum_weight[eid] = sum;
          }
          s->sum_weight[nid] = sum;
        }
    }
}

/*
  Given list of integer counts, generate real-valued weights
  and use these to flip random coin returning value between 0 and len-1
  This version only gets used in ratorder mode
*/
static inline int next_random_move(state_t *s, int r) {
    int nid = s->rat_position[r];
    int nnid = -1;
    random_t *seedp = &s->rat_seed[r];
    double tsum = compute_sum_weight(s, nid);
    graph_t *g = s->g;
    int eid;
    
    double val = next_random_float(seedp, tsum);

    double psum = 0.0;
    for (eid = g->neighbor_start[nid]; eid < g->neighbor_start[nid+1]; eid++) 
	{
		/* Node weights valid, since were computed by compute_sum_weight or earlier */
		psum += s->node_weight[g->neighbor[eid]];
		if (val < psum) {
			nnid = g->neighbor[eid];
			break;
		}
    }

    if (nnid == -1) {
	/* Shouldn't get here */
	int degree = g->neighbor_start[nid+1] - g->neighbor_start[nid];
	outmsg("Internal error.  next_random_move.  Didn't find valid move.  Node %d. Degree = %d, Target = %.2f/%.2f.  Limit = %.2f\n",
	       nid, degree, val, tsum, psum);
	nnid = 0;
    }

    return nnid;
}

/*
  Given list of increasing numbers, and target number,
  find index of first one where target is less than list value
*/

/*
  Linear search
 */
static inline int locate_value_linear(double target, double *list, int len) {
    int i;
    for (i = 0; i < len; i++)
	if (target < list[i])
	    return i;
    /* Shouldn't get here */
    return -1;
}
/*
  Binary search down to threshold, and then linear
 */
static inline int locate_value(double target, double *list, int len) {
    int left = 0;
    int right = len-1;
    while (left < right) {
	if (right-left+1 < BINARY_THRESHOLD)
	    return left + locate_value_linear(target, list+left, right-left+1);
	int mid = left + (right-left)/2;
	if (target < list[mid])
	    right = mid;
	else
	    left = mid+1;
    }
    return right;
}


/*
  Version that can be used in synchronous or batch mode, where certain that node weights are already valid.
  And have already computed sum of weights for each node, and cumulative weight for each neighbor
  Given list of integer counts, generate real-valued weights
  and use these to flip random coin returning value between 0 and len-1
*/
static inline int fast_next_random_move(state_t *s, int r) {
    int nid = s->rat_position[r];
    graph_t *g = s->g;
    random_t *seedp = &s->rat_seed[r];
    /* Guaranteed that have computed sum of weights */
    double tsum = s->sum_weight[nid];    
    double val = next_random_float(seedp, tsum);

    int estart = g->neighbor_start[nid];
    int elen = g->neighbor_start[nid+1] - estart;
    int offset = locate_value(val, &s->neighbor_accum_weight[estart], elen);
#if DEBUG
    if (offset < 0) {
	/* Shouldn't get here */
	outmsg("Internal error.  fast_next_random_move.  Didn't find valid move.  Target = %.2f/%.2f.\n",
	       val, tsum);
	return 0;
    }
#endif
    return g->neighbor[estart + offset];
}

/* Step when in synchronous mode */
static void synchronous_step(state_t *s) {
    int ri;

    find_all_sums(s);
    for (ri = 0; ri < s->nrat; ri++) {
	s->rat_position[ri] = fast_next_random_move(s, ri);
    }
    take_census(s);
    compute_all_weights(s);
}

/* Process single batch */
static inline void do_batch(state_t *s, int bstart, int bcount, int* buffer) {
    int ri, ti;
    graph_t *g = s->g;
    int nnode = g->nnode;
    // clear_delta_count(s);

#pragma omp parallel
    {
    // compute find_all_sums
    #if TIME
          double start;
    #pragma omp master
          start = currentSeconds();
    #endif
          find_all_sums(s);
          
    // compute fast_next_random_move
    #if TIME
    #pragma omp master
          s->sums_time += currentSeconds() - start;
    #pragma omp master
          start = currentSeconds();
    #endif
          
    #pragma omp for
            for (ri = 0; ri < bcount; ri++) {
              int rid = ri+bstart;
              buffer[ri] = s->rat_position[rid];
              s->rat_position[rid] = fast_next_random_move(s, rid);
    }

// update the position of rats
#if TIME
#pragma omp master
        s->moves_time += currentSeconds() - start;
#pragma omp master
#endif
      
#pragma omp for private(ri)
        for(ti = 0; ti < 12; ti++){
          int prim = (nnode + 11 / 12);
          int StartPre = prim * ti;
          int EndPre = ti == 11 ? nnode : prim * (ti + 1);
          
          for (ri = 0; ri < bcount; ri++) {
            int nnid = buffer[ri];
            if(StartPre <= nnid && nnid < EndPre){
              s->rat_count[ nnid ] -= 1;
            }
            int npid = s->rat_position[ri+bstart];
            if(StartPre <= npid && npid < EndPre){
              s->rat_count[ npid ] += 1; 
            }
          }
        }

// compute_all_weights
#if TIME
#pragma omp master
        s->updates_time += currentSeconds() - start;
#pragma omp master
      start = currentSeconds();
#endif
      
      /* Update weights */
      compute_all_weights(s);

#if TIME                        /* end compute_all_weights */
#pragma omp master
      s->weights_time += currentSeconds() - start;
#endif
    }
}

static void batch_step(state_t *s) {
    int rid = 0;
    int bsize = s->batch_size;
    int nrat = s->nrat;
    int bcount;

    int* batch_buffer = (int*) malloc( sizeof(int)*bsize );
    // init_delta_count(s);
    
    while (rid < nrat) {
	bcount = nrat - rid;
	if (bcount > bsize)
	    bcount = bsize;
	do_batch(s, rid, bcount, batch_buffer);
	rid += bcount;
    }
    free(batch_buffer);
    
}

static void ratorder_step(state_t *s) {
    int rid;
    for (rid = 0; rid < s->nrat; rid++) {
	int npid = s->rat_position[rid];
	int nnid = next_random_move(s, rid);
	s->rat_position[rid] = nnid;
	s->rat_count[npid]--;
	s->rat_count[nnid]++;
	s->node_weight[npid] = compute_weight(s, npid);
	s->node_weight[nnid] = compute_weight(s, nnid);
    }
}

typedef void (*stepper_t)(state_t *s);

double simulate(state_t *s, int count, update_t update_mode, int dinterval, bool display) {
    int i;
    /* Compute and show initial state */
    bool show_counts = true;
    stepper_t step_function;
    if (update_mode == UPDATE_SYNCHRONOUS)
	step_function = synchronous_step;
    else if (update_mode == UPDATE_BATCH)
	step_function = batch_step;
    else
	step_function = ratorder_step;

    double start = currentSeconds();
    take_census(s);
    compute_all_weights(s);
    if (display)
	show(s, show_counts);

    for (i = 0; i < count; i++) {
	step_function(s);
	if (display) {
	    show_counts = (((i+1) % dinterval) == 0) || (i == count-1);
	    show(s, show_counts);
	}
    }
    double delta = currentSeconds() - start;
    done();

#if TIME
    fprintf(stderr, "sums_time: %.5f\n", s->sums_time);
    fprintf(stderr, "moves_time: %.5f\n", s->moves_time);
    fprintf(stderr, "updates_time: %.5f\n", s->updates_time);
    fprintf(stderr, "weights_time: %.5f\n", s->weights_time);
#endif
    return delta;
}
