function simulate(pomcp::POMCPOWPlanner, h_node::POWTreeObsNode{B,A,O}, s::S, d) where {B,S,A,O}

    tree = h_node.tree
    h = h_node.node

    sol = pomcp.solver

    if POMDPs.isterminal(pomcp.problem, s) || d <= 0
        return (0.0,1)
    end

    if sol.enable_action_pw
        total_n = tree.total_n[h]
        if length(tree.tried[h]) <= sol.k_action*total_n^sol.alpha_action
            if h == 1
                a = next_action(pomcp.next_action, pomcp.problem, tree.root_belief, POWTreeObsNode(tree, h))
            else
                a = next_action(pomcp.next_action, pomcp.problem, StateBelief(tree.sr_beliefs[h]), POWTreeObsNode(tree, h))
            end
            if !sol.check_repeat_act || !haskey(tree.o_child_lookup, (h,a))
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, pomcp.problem, POWTreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, pomcp.problem, POWTreeObsNode(tree, h), a),
                            sol.check_repeat_act)
            end
        end
    else # run through all the actions
        if isempty(tree.tried[h])
            if h == 1
                action_space_iter = POMDPs.actions(pomcp.problem, tree.root_belief)
            else
                action_space_iter = POMDPs.actions(pomcp.problem, StateBelief(tree.sr_beliefs[h]))
            end
            anode = length(tree.n)
            for a in action_space_iter
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, pomcp.problem, POWTreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, pomcp.problem, POWTreeObsNode(tree, h), a),
                            false)
            end
        end
    end
    total_n = tree.total_n[h]

    best_node = select_best(pomcp.criterion, h_node, pomcp.solver.rng)
    a = tree.a_labels[best_node]

    new_node = false
    total_counts=0
    total_reward=0.0
    if tree.n_a_children[best_node] <= sol.k_observation*(tree.n[best_node]^sol.alpha_observation)

        sp, o, r = @gen(:sp, :o, :r)(pomcp.problem, s, a, sol.rng)

        if sol.check_repeat_obs && haskey(tree.a_child_lookup, (best_node,o))
            hao = tree.a_child_lookup[(best_node, o)]
        else
            new_node = true
            hao = length(tree.sr_beliefs) + 1
            push!(tree.sr_beliefs,
                  init_node_sr_belief(pomcp.node_sr_belief_updater,
                                      pomcp.problem, s, a, sp, o, r))
            """
            TODO: init_node是当o为新节点时，在创建新节点的同时把s'和w(s')加入到o中。
            因此在这我们需要进行补偿性采样，把s'和weight加入到其他的所有o中，然后在其他o中也进行补偿性采样。
            """
            """
            Step 1:把s'和weight加入到其他的所有o中.
            """   
            """
            Step 2:从任意一个其他o中，取出所有的(s',r)对中的s'，计算w，加入到现在的o的belief中
            """
            if(length(tree.generated[best_node])!=0)#如果存在其他o，执行step1.2
                first_pair=first(tree.generated[best_node])
                first_pair_hao=first_pair.second
                if(length(tree.sr_beliefs[first_pair_hao].dist.items)<1000) #step1
                    for pair in tree.generated[best_node]
                        o_temp=pair.first
                        hao_temp=pair.second
                        push_weighted!(tree.sr_beliefs[hao_temp], pomcp.node_sr_belief_updater, s, sp, r)
                    end
                end
                #step2
                for i in 1:length(tree.sr_beliefs[first_pair_hao].dist.items)#step2
                    sp_temp=tree.sr_beliefs[first_pair_hao].dist.items[i][1]
                    r_temp=tree.sr_beliefs[first_pair_hao].dist.items[i][2]
                    push_weighted!(tree.sr_beliefs[hao], pomcp.node_sr_belief_updater, s, sp_temp, r_temp)
                end
               
               
            end
           
      
            
            push!(tree.total_n, 0)
            push!(tree.tried, Int[])
            push!(tree.o_labels, o)

            if sol.check_repeat_obs
                tree.a_child_lookup[(best_node, o)] = hao
            end
            tree.n_a_children[best_node] += 1
        end
        push!(tree.generated[best_node], o=>hao)
            """
            Step3:
            对其他每个o进行补偿性采样，
            """
            for j in 1:(length(tree.generated[best_node])-1)
                pair=tree.generated[best_node][j]
                o_temp=pair.first
                hao_temp=pair.second
                before_weight=tree.sr_beliefs[hao_temp].dist.cdf[end-1]
                before_sample_times=tree.total_n[hao_temp]
                new_added_weight=tree.sr_beliefs[hao_temp].dist.cdf[end]-tree.sr_beliefs[hao_temp].dist.cdf[end-1]
                cs_times=div(before_sample_times*new_added_weight,before_weight)#下取整
                if (cs_times!=0)
                    for i in 1:cs_times
                        R_temp,counts_temp=simulate(pomcp, POWTreeObsNode(tree, hao_temp), sp, d-1)
                        R = r + POMDPs.discount(pomcp.problem)*R_temp
                        total_counts+=counts_temp
                        total_reward+=R*counts_temp
                    end
                end
            end
    else

        sp, r = @gen(:sp, :r)(pomcp.problem, s, a, sol.rng)

    end

    if r == Inf
        @warn("POMCPOW: +Inf reward. This is not recommended and may cause future errors.")
    end

    if new_node
        R = r + POMDPs.discount(pomcp.problem)*estimate_value(pomcp.solved_estimate, pomcp.problem, sp, POWTreeObsNode(tree, hao), d-1)
        total_counts+=1
        total_reward+=R
        """
        total_counts++
        total_reward+=R
        """
    else
        pair = rand(sol.rng, tree.generated[best_node])
        o = pair.first
        hao = pair.second
        """
        To-do: 当o不是新节点时，需要把这轮从G中得到的s'，加入到当前a下的所有o分支的B(hao)中，同时根据每个分支的o计算相应的权重
        然后在每个O分支下，对新粒子进行补偿性采样。
        """
        """
        Step 1:把s'和weight加入到所有o中,包括自己.
        """
        if(length(tree.sr_beliefs[hao].dist.items)<1000) #限制粒子数为1000
            for pair in tree.generated[best_node]
                o_temp=pair.first
                hao_temp=pair.second     
                push_weighted!(tree.sr_beliefs[hao_temp], pomcp.node_sr_belief_updater, s, sp, r)
            end
            """
            Step2:补偿性采样
            """
             for pair in tree.generated[best_node]
                 o_temp=pair.first
                 hao_temp=pair.second
                 before_weight=tree.sr_beliefs[hao_temp].dist.cdf[end-1]
                 before_sample_times=tree.total_n[hao_temp]
                 new_added_weight=tree.sr_beliefs[hao_temp].dist.cdf[end]-tree.sr_beliefs[hao_temp].dist.cdf[end-1]
                 cs_times=div(before_sample_times*new_added_weight,before_weight)#下取整
                 if (cs_times!=0)
                     for i in 1:cs_times
                         R_temp,counts_temp=simulate(pomcp, POWTreeObsNode(tree, hao_temp), sp, d-1)            
                         R = r + POMDPs.discount(pomcp.problem)*R_temp
                         total_reward+=R*counts_temp
                         total_counts+=counts_temp
                     end
                 end
             end
        end
        # push_weighted!(tree.sr_beliefs[hao], pomcp.node_sr_belief_updater, s, sp, r)#默认把一个粒子加入到当前o
      
       """
       Step3:随机采样
      
       """
       sp, r = rand(sol.rng, tree.sr_beliefs[hao])
       R_temp,counts_temp=simulate(pomcp, POWTreeObsNode(tree, hao), sp, d-1)

       R = r + POMDPs.discount(pomcp.problem)*R_temp
       total_counts+=counts_temp
       total_reward+=R*counts_temp
        # sp, r = rand(sol.rng, tree.sr_beliefs[hao])
        # R = r + POMDPs.discount(pomcp.problem)*simulate(pomcp, POWTreeObsNode(tree, hao), sp, d-1)
    end

    """
    返回Reward和次数
    """
    tree.n[best_node] += total_counts
    tree.total_n[h] += total_counts
    if tree.v[best_node] != -Inf
        tree.v[best_node] = ((tree.n[best_node]-total_counts)*tree.v[best_node]+total_reward)/tree.n[best_node]
    end
  
    return (total_reward/total_counts,total_counts)

    # tree.n[best_node] += 1
    # tree.total_n[h] += 1
    # if tree.v[best_node] != -Inf
    #     tree.v[best_node] += (R-tree.v[best_node])/tree.n[best_node]
    # end  
    # return R
end

