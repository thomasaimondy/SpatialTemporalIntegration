# %load input_ff_rec_transform_nengo_ocl.py
# (c) Sep 2015 Aditya Gilra, EPFL.

"""
learning of arbitrary feed-forward or recurrent transforms
in Nengo simulator
written by Aditya Gilra (c) Sep 2015.
"""
# these give import warning, import them before rate_evolve which converts warnings to errors
import input_rec_transform_nengo_plot as myplot
OCL = True
import nengo_ocl
import nengo
## Note that rate_evolve.py converts warnings to errors
## so import nengo first as importing nengo generates
##  UserWarning: IPython.nbformat.current is deprecated.
from rate_evolve import *
import warnings                         # later I've set the warnings to errors off.
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import shelve, contextlib
from os.path import isfile
import os,sys

########################
### Constants/parameters
########################

###
### Overall parameter control ###
###
errorLearning = True                    # obsolete, leave it True; error-based PES learning OR algorithmic
recurrentLearning = True                # obsolete, leave it True; learning on ff+rec both
plastDecoders = False                   # whether to just have plastic decoders or plastic weights
inhibition = False#True and not plastDecoders # clip ratorOut weights to +ve only and have inh interneurons

learnIfNoInput = False                  # obsolete, leave False; Learn only when input is off (so learning only on error current)
errorFeedback = True                    # Forcefeed the error into the network (used only if errorLearning)
learnFunction = True                    # obsolete, leave True; whether to learn a non-linear function or a linear matrix
funcType = 'vanderPol'                  # if learnFunction, then vanderPol oscillator

initLearned = False and recurrentLearning and not inhibition
                                        # whether to start with learned weights (bidirectional/unclipped)
                                        # currently implemented only for recurrent learning
fffType = ''                            # whether feedforward function is linear or non-linear
testLearned = False                     # whether to test the learning, uses weights from continueLearning, but doesn't save again.
continueLearning = False             # whether to load old weights and continue learning from there
                                        # saving weights at the end is always enabled
# for both testLearned and continueLearned, set testLearnedOn below:

testLearnedOn = '_trials_seed2by50.0amplVaryHeightsScaled'  # for vanderPol

saveSpikes = True                       # save the spikes if testLearned and saveSpikes
if funcType == 'vanderPol' and not testLearned:
    trialClamp = True                   # whether to reset reference and network into trials during learning (or testing if testLearned)
else:
    trialClamp = False                  # whether to reset reference and network into trials during learning (or testing if testLearned)
copycatLayer = False                    # use copycat layer OR odeint rate_evolve
copycatPreLearned = copycatLayer        # load pre-learned weights into InEtoEexpect & EtoEexpect connections OR Wdesired function
                                        # system doesn't learn with Wdesired function i.e. (copycatLayer and not copycatPreLearned), FF connection mismatch issue possibly
copycatWeightsFile = 'data/ff_ocl_Nexc3000_noinptau_seeds2344_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_vanderPol_trials_seed2by50.0amplVaryHeightsScaled_10000.0s_endweights.shelve'
zeroLowWeights = False                  # set to zero weights below a certain value
weightErrorCutoff = 0.                  # Do not pass any abs(error) for weight change below this value
randomInitWeights = False#True and not plastDecoders and not inhibition
                                        # start from random initial weights instead of zeros
                                        # works only for weights, not decoders as decoders are calculated from transform
randomWeightSD = 1e-4                   # old - perhaps approx SD of weight distribution EtoE for linear, for InEtoE, Wdyn2/20 is used
weightRegularize = False                # include a weight decay term to regularize weights
randomDecodersType = ''                 # choose one of these
randomDecodersFactor = 0.625              # x instead of 1, in x+random
randomDecodersAmount = 2.               # the relative randomness to 1, in 1plusrandom
sparseWeights = False
sparsityFake = 0.15                     # CAUTION: this is only for filename; actual value is set in nengo/builder/connection.py

# very important to set dynNoise to not None if using dynNoiseMean
dynNoise = None                         # no noise in ff and rec neuronal ensembles
dynNoiseMean = None                     # no noise in ff and rec neuronal ensembles


# very important to set errNoise to not None if using errNoiseMean
errNoise = None                         # no noise in error signal
errNoiseMean = None                     # no noise in error signal

###
### Nengo model params ###
###
seedR0 = 2              # seed set while defining the Nengo model
seedR1 = 3              # another seed for the first layer
                        # some seeds give just flat lines for Lorenz! Why?
seedR2 = 4              # another seed for the second layer
                        # this is just for reproducibility
                        # seed for the W file is in rate_evolve.py
                        # output is very sensitive to this seedR
                        # as I possibly don't have enough neurons
                        # to tile the input properly (obsolete -- for high dim)
seedR4 = 4              # for the nengonetexpect layer to generate reference signal

seedRin = 2
np.random.seed([seedRin])# this seed generates the inpfn below (and non-nengo anything random)
          
tau = 0.02              # second # as for the rate network
#tau = 0.1               # second
                        # original is 0.02, but 0.1 gives longer response
tau_AMPA = 1e-3         # second # fast E to I connections

spikingNeurons = False  # whether to use Ensemble (LIF neurons) or just Node
                        #  the L2 has to be neurons to apply PES learning rule,
                        #  rest can be Ensemble or Node
if spikingNeurons:
    neuronType = nengo.neurons.LIF()
                        # use LIF neurons for all ensembles
else:
    neuronType = None  
                       

###
### choose dynamics evolution matrix ###
###

init_vec_idx = 0        # first / largest response vector
evolve = 'fixedW'       # fixed W: Schaub et al 2015 / 2D oscillator
evolve_dirn = 'arb'     # arbitrary normalized initial direction
inputType = 'amplVaryHeightsScaled'

###
### Load dynamics evolution matrix and stimulus 'direction'
###
M,W,Winit,lambdas,a0s,desc_str = loadW(evolve)
v,w,dir_str = get_relevant_modes(evolve_dirn,W,lambdas,a0s)
y0,y01,y02 = get_stim_dirn(evolve_dirn,v,w,init_vec_idx,W)


N = len(v)
B = 2 * (y0-dot(W,y0))                                  # (I-W)y0, arb:2
rampT = 0.25#0.1                                        # second
dt = 0.001                                              # second

###
### recurrent and feedforward connection matrices ###
###
if errorLearning:                                       # PES plasticity on
    Tmax = 500.                                       # second
    continueTmax = 500.                               # if continueLearning or testLearned,
                                                        #  then start with weights from continueTmax                                                     #  and run the learning/testing for Tmax
    reprRadius = 1.                                     # neurons represent (-reprRadius,+reprRadius)
    if recurrentLearning:                               # L2 recurrent learning
        #PES_learning_rate = 9e-1                        # learning rate with excPES_integralTau = Tperiod
        #                                                #  as deltaW actually becomes very small integrated over a cycle!
        if testLearned:
            eta = 1e-15                                 # effectively no learning
        else:
            eta = 1e-4                                  # this learning rate doesn't change weights much over 2s
        PES_learning_rate_FF = eta
        PES_learning_rate_rec = eta 
        if learnFunction:
            if funcType == 'vanderPol':
                # van der Pol oscillator (relaxation oscillator for mu>>1)
                Nexc = 300                                 # number of excitatory neurons
                mu,scale,taudyn = 2.,1.,0.125#0.25
                Wdesired = lambda x: np.array([x[1],mu*(1-scale*x[0]**2)*x[1]-x[0]])\
                                        /taudyn*tau + x
                                                            # scaled van der pol equation, using y=dx/dt form
                                                            #  note the extra tau and 'identity' matrix
                reprRadius = 5.                             # neurons represent (-reprRadius,+reprRadius)
                reprRadiusIn = 0.2                          # ratorIn is lower than ratorOut since ratorOut integrates the input
                Tperiod = 4.                                # second
                rampT = 0.1                                 # second (shorter ramp for van der Pol)
                B /= 3.                                     # reduce input, else longer sharp "down-spike" in reference cannot be reproduced
                inputreduction = 50.0                      # input reduction factor
            elif funcType == 'Lorenz':
                # Lorenz attractor (chaos in continuous dynamics needs at least 3-dim and non-linearity)
                Nexc = 5000                                 # number of excitatory neurons
                                                            # 200*N was not good enough for Lorenz attractor
                N = 3                                       # number of dimensions of dynamical system
                taudyn = 1.                                 # second
                reprRadius = 30.                            # neurons represent (-reprRadius,+reprRadius)
                reprRadiusIn = reprRadius/5.                # ratorIn is lower than ratorOut since ratorOut integrates the input
                # https://pythonhosted.org/nengo/examples/lorenz_attractor.html
                # in the above nengo example, they've transformed x[2]'=x[2]-28 to get a zero baseline for x[2]
                Wdesired = lambda x: np.array( (10.*(x[1]-x[0]), -x[0]*x[2]-x[1], x[0]*x[1]-8./3.*(x[2]+28.) ) )/taudyn*tau + x
                                                            #  note the extra tau and 'identity matrix'
                B = append(B,0.)
                if testLearned:
                    Tperiod = 100.                          # second per ramp-release-evolve cycle, NA for Lorenz
                else:
                    Tperiod = 20.                           # second per ramp-release-evolve cycle, NA for Lorenz
                                                            # Also Tnolearning = 5*Tperiod
                inputreduction = 10.0                        # input reduction factor
            elif funcType == 'LinOsc':
                reprRadius = 1.0
                reprRadiusIn = reprRadius/5.                # ratorIn is lower than ratorOut since ratorOut integrates the input
                # linear oscillator put in as a function
                Nexc = 2000                                 # number of excitatory neurons
                taudyn = 0.1                                # second
                decay_fac = -0.2                            # effective decay_tau = taudyn/decay_fac
                Wdesired = lambda x: np.array( [decay_fac*x[0]-x[1], x[0]+decay_fac*x[1]] )/taudyn*tau + x
                                                            #  note the extra tau and 'identity matrix'
                Tperiod = 2.                                # second per ramp-release-evolve cycle
                rampT = 0.1                                 # second (shorter ramp for lin osc)
                inputreduction = 8.0                        # input reduction factor
            elif funcType == 'robot1':
                reprRadius = 6.0
                reprRadiusIn = reprRadius/5.                # ratorIn is lower than ratorOut since ratorOut integrates the input
                # one-link robot arm with angle blocked at +-0.6
                N = 2
                Nexc = 2000                                 # number of excitatory neurons
            else:
                print("Specify a function type")
                sys.exit(1)
        else:
            Nexc = 1000                                 # number of excitatory neurons
            Wdesired = W
            Tperiod = 2.                                # second per ramp-release-evolve cycle
        Wdyn1 = np.zeros(shape=(N,N))
        if plastDecoders:                               # only decoders are plastic ??
            Wdyn2 = np.zeros(shape=(N,N))
        else:                                           # weights are plastic, connection is now between neurons
            if randomInitWeights:
                Wdyn2 = np.random.normal(size=(Nexc,Nexc))*randomWeightSD
            else:
                Wdyn2 = np.zeros(shape=(Nexc,Nexc))
        #Wdyn2 = W
        #Wdyn2 = W+np.random.randn(N,N)*np.max(W)/5.
        W = Wdesired                                    # used by matevolve2 below
        Wtransfer = np.eye(N)

Nerror = 200*N                                          # number of error calculating neurons
reprRadiusErr = 0.2                                     # with error feedback, error is quite small

###
### time params ###
###
weightdt = Tmax/20.                                     # how often to probe/sample weights
Tclamp = 0.5                                            # time to clamp the ref, learner and inputs after each trial (Tperiod)
Tnolearning = 4*Tperiod
                                                        # in last Tnolearning s, turn off learning & weight decay

if Nexc > 3000: saveWeightsEvolution = False
else: saveWeightsEvolution = True                       # don't save for Nexc~4000 else too much disk space,
                                                        #  and also shelve crashes (need hdf5); GPU (OCL) runs out of mem

###
### Generate inputs for L1 ###
###
zerosN = np.zeros(N)

if inputType == 'amplVaryHeightsScaled':
    heights = np.random.normal(size=(N,int(Tmax/Tperiod)+1))
    heights = heights/np.linalg.norm(heights,axis=0)
    if funcType == 'vanderPol': heights[0,:]/=3.
    ## random uniform 'white-noise' with 10 ms steps interpolated by cubic
    ##  50ms is longer than spiking-network response-time, and assumed shorter than tau-s of the dynamical system.
    noisedt = 50e-3
    # cubic interpolation for long sim time takes up ~64GB RAM and then hangs, so linear or nearest interpolation.
    noiseN = np.random.uniform(-reprRadiusIn,reprRadiusIn,size=(N,int(Tmax/noisedt)+1))
    if funcType == 'vanderPol': noiseN[0,:]/=3.
    noisefunc = interp1d(np.linspace(0,Tmax,int(Tmax/noisedt)+1),noiseN,kind='nearest',\
                                            bounds_error=False,fill_value=0.,axis=1)
    del noiseN
    if trialClamp:
        inpfn = lambda t: noisefunc(t)/2. * np.float((t%Tperiod)<(Tperiod-Tclamp)) + \
                            ( heights[:,int(t/Tperiod)]*reprRadiusIn )/2.
    else:
        inpfn = lambda t: ( noisefunc(t) + heights[:,int(t/Tperiod)]*reprRadiusIn )/2.

else:
    inpfn = lambda t: 0.0*np.ones(N)*reprRadius          # constant input, currently zero
    
# nengo_ocl and odeint generate some warnings, so we reverse the 'warnings to errors' from rate_evolve.py
warnings.filterwarnings('ignore') 

###
### Reference evolution used when copycat layer is not used for reference ###
###
if Wdesired.__class__.__name__=='function':
    def matevolve2(y,t):
        inpfnVal = inpfn(t)/tau
        return ( ((Wdesired(y)-y)/tau + inpfnVal) if (t%Tperiod)<(Tperiod-Tclamp) else -y/dt )
                                                        # on integration, 'rampLeave' becomes B*t/rampT*reprRadius
                                                        # for Tclamp at the end of the trial, clamp the output to zero
else:
    eyeN = np.eye(N)
    def matevolve2(y,t):
        return dot((Wdesired-eyeN)/tau,y) + inpfn(t)/tau
                                                        # on integration, 'rampLeave' becomes B*t/rampT*reprRadius
trange = arange(0.0,Tmax,dt)
y = odeint(matevolve2,0.001*np.ones(N),trange,hmax=dt)  # set hmax=dt, to avoid adaptive step size
                                                        # some systems (van der pol) have origin as a fixed pt
                                                        # hence start just off-origin
rateEvolveFn = interp1d(trange,y,axis=0,kind='linear',\
                        bounds_error=False,fill_value=0.)
                                                        # used for the error signal below
del y                                                   # free some memory

if errorLearning:
    if not weightRegularize:
        excPES_weightsDecayRate = 0.        # no decay of PES plastic weights
    else:
        excPES_weightsDecayRate = 1./1e1#1./1e4    # 1/tau of synaptic weight decay for PES plasticity 
        
    excPES_integralTau = None               # don't integrate deltaW for PES plasticity (default) 
                                            #  for generating the expected response signal for error computation
    errorAverage = False                    # whether to average error over the Tperiod scale
                                            # Nopes, this won't make it learn the intricate dynamics
    
    errorFeedbackGain = 10.                 # Feedback gain
    weightErrorTau = 10*tau                 # filter the error to the PES weight update rule
    errorFeedbackTau = 1*tau                # synaptic tau for the error signal into layer2.ratorOut
                                            # even if copycatLayer is True, you need this filtering, else too noisy spiking and cannot learn
    errorGainDecay = False                  # whether errorFeedbackGain should decay exponentially to zero
                                            # decaying gain gives large weights increase below some critical gain ~3
    errorGainDecayRate = 1./200.            # 1/tau for decay of errorFeedbackGain if errorGainDecay is True
    errorGainProportion = False             # scale gain proportionally to a long-time averaged |error|
    errorGainProportionTau = Tperiod        # time scale to average error for calculating feedback gain
    weightsFileName = "data/gilra/tmp/learnedWeights.shelve"
if errorLearning and recurrentLearning:
    inhVSG_weightsDecayRate = 1./40.
else:
    inhVSG_weightsDecayRate = 1./2.         # 1/tau of synaptic weight decay for VSG plasticity


pathprefix = '/data1/wanghuaijin/hjw/FOLLOW/data/'
inputStr = ('_trials' if trialClamp else '') + \
        ('_seed'+str(seedRin)+'by'+str(inputreduction)+inputType if inputType != 'rampLeave' else '')
baseFileName = pathprefix+'ff'+('_ocl' if OCL else '')+'_Nexc'+str(Nexc) + \
                    '_seeds'+str(seedR0)+str(seedR1)+str(seedR2)+str(seedR4) + \
                    fffType + \
                    ('' if dynNoise is None else '_dynnoiz'+str(dynNoise)) + \
                    ('' if dynNoiseMean is None else '_dynnoizmean'+str(dynNoiseMean)) + \
                    ('' if errNoise is None else '_errnoiz'+str(errNoise)) + \
                    ('' if errNoiseMean is None else '_errnoizmean'+str(errNoiseMean)) + \
                    ('_inhibition' if inhibition else '') + \
                    ('_zeroLowWeights' if zeroLowWeights else '') + \
                    '_weightErrorCutoff'+str(weightErrorCutoff) + \
                    ('_randomInitWeights'+str(randomWeightSD) if randomInitWeights else '') + \
                    ('_weightRegularize'+str(excPES_weightsDecayRate) if weightRegularize else '') + \
                    '_nodeerr' + ('_plastDecoders' if plastDecoders else '') + \
                    (   (   '_learn' + \
                            ('_rec' if recurrentLearning else '_ff') + \
                            ('' if errorFeedback else '_noErrFB') \
                        ) if errorLearning else '_algo' ) + \
                    ('_initLearned' if initLearned else '') + \
                    (randomDecodersType + (str(randomDecodersAmount)+'_'+str(randomDecodersFactor)\
                                            if 'plusrandom' in randomDecodersType else '')) + \
                    ('_sparsity'+str(sparsityFake) if sparseWeights else '') + \
                    ('_learnIfNoInput' if learnIfNoInput else '') + \
                    (('_precopy' if copycatPreLearned else '') if copycatLayer else '_nocopycat') + \
                    ('_func_'+funcType if learnFunction else '') + \
                    (testLearnedOn if (testLearned or continueLearning) else inputStr)
                        # filename to save simulation data
dataFileName = baseFileName + \
                    ('_continueFrom'+str(continueTmax)+inputStr if continueLearning else '') + \
                    ('_testFrom'+str(continueTmax)+inputStr if testLearned else '') + \
                    '_'+str(Tmax)+'s'
print('data will be saved to', dataFileName, '_<start|end|currentweights>.shelve')
if continueLearning or testLearned:
    weightsSaveFileName = baseFileName + '_'+str(continueTmax+Tmax)+'s_endweights.shelve'
    weightsLoadFileName = baseFileName + '_'+str(continueTmax)+'s_endweights.shelve'
else:
    weightsSaveFileName = baseFileName + '_'+str(Tmax)+'s_endweights.shelve'
    weightsLoadFileName = baseFileName + '_'+str(Tmax)+'s_endweights.shelve'    

if __name__ == "__main__":
    #########################
    ### Create Nengo network
    #########################
    print('building model')
    mainModel = nengo.Network(label="Single layer network", seed=seedR0)
    with mainModel:
        nodeIn = nengo.Node( size_in=N, output = lambda timeval,currval: inpfn(timeval) )
        ratorIn = nengo.Ensemble( Nexc, dimensions=N, radius=reprRadiusIn,
                            neuron_type=nengo.neurons.LIF(),
                            max_rates=nengo.dists.Uniform(200, 400),
                            noise=None, seed=seedR1, label='ratorIn' )
        nengo.Connection(nodeIn, ratorIn, synapse=None)         # No filtering here as no filtering/delay in the plant/arm
        # another layer with learning incorporated
        ratorOut = nengo.Ensemble( Nexc, dimensions=N, radius=reprRadius,
                            neuron_type=nengo.neurons.LIF(),
                            #bias=nengo.dists.Uniform(-nrngain,nrngain), gain=np.ones(Nexc)*nrngain, 
                            max_rates=nengo.dists.Uniform(200, 400),
                            noise=None, seed=seedR2, label='ratorOut' )
        
        if trialClamp:
            # clamp ratorIn and ratorOut at the end of each trial (Tperiod) for 100ms.
            #  Error clamped below during end of the trial for 100ms.
            clampValsZeros = np.zeros(Nexc)
            clampValsNegs = -100.*np.ones(Nexc)
            endTrialClamp = nengo.Node(lambda t: clampValsZeros if (t%Tperiod)<(Tperiod-Tclamp) else clampValsNegs)
            nengo.Connection(endTrialClamp,ratorIn.neurons,synapse=1e-3)
            nengo.Connection(endTrialClamp,ratorOut.neurons,synapse=1e-3)
                                                                # fast synapse for fast-reacting clamp
        
        if not plastDecoders:
            # If initLearned==True, these weights will be reset below using InEtoEfake and EtoEfake

            InEtoE = nengo.Connection(ratorIn.neurons, ratorOut.neurons,
                                            transform=Wdyn2/20., synapse=tau)
                                                                # Wdyn2 same as for EtoE, but mean(InEtoE) = mean(EtoE)/20
            EtoE = nengo.Connection(ratorOut.neurons, ratorOut.neurons,
                                transform=Wdyn2, synapse=tau)   # synapse is tau_syn for filtering

        # initLearned
        # probes
        nodeIn_probe = nengo.Probe(nodeIn, synapse=None)
        ratorIn_probe = nengo.Probe(ratorIn, synapse=tau)
        # don't probe what is encoded in ratorIn, rather what is sent to ratorOut
        # 'output' reads out the output of the connection InEtoE in nengo 2.2.1.dev0
        #  but in older nengo ~2.0, the full variable encoded in ratorOut (the post-ensemble of InEtoE)
        # NOTE: InEtoE is from neurons to neurons, so 'output' is Nexc-dim not N-dim!
        #ratorIn_probe = nengo.Probe(InEtoE, 'output')
        #ratorIn_probe = nengo.Probe(InEtoE, 'input', synapse=tau)
        # don't probe ratorOut here as this calls build_decoders() separately for this;
        #  just call build_decoders() once for ratorOut2error, and probe 'output' of that connection below
        #ratorOut_probe = nengo.Probe(ratorOut, synapse=tau)
                                                                # synapse is tau for filtering
                                                                # Default is no filtering
       
                                                                # this becomes too big for shelve (ndarray.dump())
                                                                #  for my Lorenz _end simulation of 100s
                                                                #  gives SystemError: error return without exception set
                                                                # use python3.3+ or break into smaller sizes
                                                                # even with python3.4, TypeError: gdbm mappings have byte or string elements only

    ############################
    ### Learn ratorOut EtoE connection
    ############################
    with mainModel:
        if errorLearning:
            ###
            ### copycat layer only for recurrent learning ###
            ###
            # another layer that produces the expected signal for above layer to learn
            # force the encoders, maxrates and intercepts to be same as ratorOut
            #  so that the weights are directly comparable between netExpect (copycat) and net2
            # if Wdesired is a function, then this has to be LIF layer
            
            ###
            ### error ensemble, could be with error averaging, gets post connection ###
            ###
            
            error = nengo.Node( size_in=N, output = lambda timeval,err: err)
            if trialClamp:
                errorOff = nengo.Node( size_in=N, output = lambda timeval,err: \
                                            err if (Tperiod<timeval<(Tmax-Tnolearning) and (timeval%Tperiod)<Tperiod-Tclamp) \
                                            else np.zeros(N), label='errorOff' )
            
            error2errorOff = nengo.Connection(error,errorOff,synapse=None)
            # Error = post - pre * desired_transform
            ratorOut2error = nengo.Connection(ratorOut,error,synapse=tau)
                                                            # post input to error ensemble (pre below)
                                                            # tau-filtered output (here) is matched to the unfiltered reference (pre below)
            # important to probe only ratorOut2error as output, and not directly ratorOut, to accommodate randomDecodersType != ''
            # 'output' reads out the output of the connection in nengo 2.2 on
            ratorOut_probe = nengo.Probe(ratorOut2error, 'output')# pred
               

            

            ###
            ### Add the relevant pre signal to the error ensemble ###
            ###
            if recurrentLearning:                           # L2 rec learning
                
                                                                # - desired output here (post above)
                                                                # tau-filtered expectOut must be compared to tau-filtered ratorOut (post above)
                
                rateEvolve = nengo.Node(rateEvolveFn)
                # Error = post - desired_output
                rateEvolve2error = nengo.Connection(rateEvolve,error,synapse=tau,transform=-np.eye(N))
                #rateEvolve2error = nengo.Connection(rateEvolve,error,synapse=None,transform=-np.eye(N))
                                                            # - desired output here (post above)
                                                            # unfiltered non-spiking reference is compared to tau-filtered spiking ratorOut (post above)
                plasticConnEE = EtoE
                rateEvolve_probe = nengo.Probe(rateEvolve2error, 'output')
                                                                # save the filtered/unfiltered reference as this is the 'actual' reference

            ###
            ### Add the exc learning rules to the connection, and the error ensemble to the learning rule ###
            ###
            EtoERulesDict = { 'PES' : nengo.PES(learning_rate=PES_learning_rate_rec,
                                            pre_tau=tau) }
            plasticConnEE.learning_rule_type = EtoERulesDict
            #plasticConnEE.learning_rule['PES'].learning_rate=0
                                                            # learning_rate has no effect
                                                            # set to zero, yet works fine!
                                                            # It works only if you set it
                                                            # in the constructor PES() above
                                                           # error to errorWt ensemble, filter for weight learning
            if not learnIfNoInput:
                # if trialClamp just forcing error to zero doesn't help, as errorWt decays at long errorWeightTau,
                #  so force errorWt also to zero, so that learning is shutoff at the end of a trial
                if trialClamp:
                    errorWt = nengo.Node( size_in=N, output = lambda timeval,errWt: \
                                                                ( errWt*(np.abs(errWt)>weightErrorCutoff) \
                                                                    if (timeval%Tperiod)<Tperiod-Tclamp else zerosN ) )
                
                nengo.Connection(errorOff,errorWt,synapse=weightErrorTau)
                                                                # error to errorWt ensemble, filter for weight learning

            error_conn = nengo.Connection(\
                    errorWt,plasticConnEE.learning_rule['PES'],synapse=dt)
            # feedforward error connection to learning rule
            if not (copycatLayer and not copycatPreLearned):    # don't learn ff weights if copycatLayer and not copycatPreLearned
                # feedforward learning rule
                InEtoERulesDict = { 'PES' : nengo.PES(learning_rate=PES_learning_rate_FF,
                                                pre_tau=tau) }
                InEtoE.learning_rule_type = InEtoERulesDict
                nengo.Connection(\
                        errorWt,InEtoE.learning_rule['PES'],synapse=dt)

            ###
            ### feed the error back to force output to follow the input (for both recurrent and feedforward learning) ###
            ###
            if errorFeedback and not testLearned:                       # no error feedback if testing learned weights
                #np.random.seed(1)
                if not errorGainProportion: # default error feedback
                    errorFeedbackConn = nengo.Connection(errorOff,ratorOut,\
                                synapse=errorFeedbackTau,\
                                transform=-errorFeedbackGain)#*(np.random.uniform(-0.1,0.1,size=(N,N))+np.eye(N)))
                
                
        
            ###
            ### error and weight probes ###
            ###
            errorOn_p = nengo.Probe(error, synapse=None, label='errorOn')
            error_p = nengo.Probe(errorWt, synapse=None, label='error')
            if saveWeightsEvolution:
                learnedWeightsProbe = nengo.Probe(\
                            plasticConnEE,'weights',sample_every=weightdt,label='EEweights')
                # feedforward weights probe
                learnedInWeightsProbe = nengo.Probe(\
                            InEtoE,'weights',sample_every=weightdt,label='InEEweights')

    if initLearned:
        pass
        ## if not initLearned, we don't care about matching weights to ideal
        ## this reduces a large set of connections, esp if Nexc is large
        #model.connections.remove(EtoEfake)
    
    #################################
    ### Run Nengo network
    #################################
    if OCL: sim = nengo_ocl.Simulator(mainModel,dt)
    else:  sim = nengo.Simulator(mainModel,dt)
    Eencoders = sim.data[ratorOut].encoders
    # randomize decoders
    


    #################################
    ### important to initialize weights before,
    ### so that they can be overridden after if continueLearning or testLearned
    #################################
    

    #################################
    ### load previously learned weights, if requested and file exists
    #################################
    if errorLearning and (continueLearning or testLearned) and isfile(weightsLoadFileName):
        print('loading previously learned weights for ratorOut from',weightsLoadFileName)
        with contextlib.closing(
                shelve.open(weightsLoadFileName, 'r', protocol=-1)
                ) as weights_dict:
            #sim.data[plasticConnEE].weights = weights_dict['learnedWeights']       # can't be set, only read
            sim.signals[ sim.model.sig[plasticConnEE]['weights'] ] \
                                = weights_dict['learnedWeights']                    # can be set if weights/decoders are set earlier
            sim.signals[ sim.model.sig[InEtoE]['weights'] ] \
                                = weights_dict['learnedWeightsIn']                  # can be set if weights/decoders are set earlier
    else:
        if continueLearning or testLearned: print('File ',weightsLoadFileName,' not found.')
        print('Not loading any pre-learned weights for ratorOut.')

    #################################
    ### load previously learned weights for the copycat layer
    #################################
    

    #################################
    ### save the expected weights
    #################################
    

    def changeLearningRate(changeFactor):
        '''
         Call this function to change the learning rate.
         Doesn't actually change learning rate, only the decay factor in the operations!
         Will only work if excPESIntegralTau = None, else the error is accumulated every step with this factor!
        '''
        # change the effective learning rate by changing the decay factor for the learning operators in the simulator
        for op in sim._step_order:
            if op.__class__.__name__=='ElementwiseInc':
                if op.tag=='PES:Inc Delta':
                    print('setting learning rate ',changeFactor,'x in',op.__class__.__name__,op.tag)
                    op.decay_factor *= changeFactor         # change learning rate for PES rule
                                                            #  PES rule doesn't have a SimPES() operator;
                                                            #  it uses an ElementwiseInc to calculate Delta
                elif op.tag=='weights += delta':
                    print('setting weight decay = 1.0 in',op.__class__.__name__,op.tag)
                    op.decay_factor = 1.0                   # no weight decay for all learning rules

        # rebuild steps (resets ops with their own state, like Processes)
        # copied from simulator.py Simulator.reset()
        sim.rng = np.random.RandomState(sim.seed)
        sim._steps = [op.make_step(sim.signals, sim.dt, sim.rng)
                            for op in sim._step_order]

    def turn_off_learning():
        '''
         Call this function to turn learning off at the end
        '''
        # set the learning rate to zero for the learning operators in the simulator
        for op in sim._step_order:
            if op.__class__.__name__=='ElementwiseInc':
                if op.tag=='PES:Inc Delta':
                    print('setting learning rate = 0.0 in',op.__class__.__name__,op.tag)
                    op.decay_factor = 0.0                   # zero learning rate for PES rule
                                                            #  PES rule doesn't have a SimPES() operator;
                                                            #  it uses an ElementwiseInc to calculate Delta
                elif op.tag=='weights += delta':
                    print('setting weight decay = 1.0 in',op.__class__.__name__,op.tag)
                    op.decay_factor = 1.0                   # no weight decay for all learning rules

        # rebuild steps (resets ops with their own state, like Processes)
        # copied from simulator.py Simulator.reset()
        sim.rng = np.random.RandomState(sim.seed)
        sim._steps = [op.make_step(sim.signals, sim.dt, sim.rng)
                            for op in sim._step_order]

    def save_data(endTag):
        #print 'pickling data'
        #pickle.dump( data_dict, open( "/lcncluster/gilra/tmp/rec_learn_data.pickle", "wb" ) )
        print('shelving data',endTag)
        # with statement causes close() at the end, else must call close() explictly
        # 'c' opens for read and write, creating it if not existing
        # protocol = -1 uses the highest protocol (currently 2) which is binary,
        #  default protocol=0 is ascii and gives `ValueError: insecure string pickle` on loading
        with contextlib.closing(
                shelve.open(dataFileName+endTag+'.shelve', 'c', protocol=-1)
                ) as data_dict:
            data_dict['trange'] = sim.trange()
            data_dict['Tmax'] = Tmax
            data_dict['rampT'] = rampT
            data_dict['Tperiod'] = Tperiod
            data_dict['dt'] = dt
            data_dict['tau'] = tau
            data_dict['nodeIn'] = sim.data[nodeIn_probe]
            data_dict['ratorOut'] = sim.data[ratorIn_probe]
            data_dict['ratorOut2'] = sim.data[ratorOut_probe]
            data_dict['errorLearning'] = errorLearning
            data_dict['spikingNeurons'] = spikingNeurons
            if testLearned and saveSpikes:
                data_dict['EspikesOut2'] = sim.data[ratorOut_EspikesOut]
                if copycatLayer:
                    data_dict['ExpectSpikesOut'] = sim.data[expectOut_spikesOut]
            if spikingNeurons:
                data_dict['EVmOut'] = sim.data[EVmOut]
                data_dict['EIn'] = sim.data[EIn]
                data_dict['EOut'] = sim.data[EOut]
            data_dict['rateEvolve'] = rateEvolveFn(sim.trange())
            if errorLearning:
                data_dict['recurrentLearning'] = recurrentLearning
                data_dict['error'] = sim.data[errorOn_p]
                data_dict['error_p'] = sim.data[error_p]
                #data_dict['learnedExcOut'] = sim.data[learnedExcOutProbe],
                #data_dict['learnedInhOut'] = sim.data[learnedInhOutProbe],
                data_dict['copycatLayer'] = copycatLayer
                data_dict['copycatPreLearned'] = copycatPreLearned
                data_dict['copycatWeightsFile'] = copycatWeightsFile
                if recurrentLearning:
                    data_dict['rateEvolveFiltered'] = sim.data[rateEvolve_probe]
                    if copycatLayer:
                        data_dict['yExpectRatorOut'] = sim.data[expectOut_probe]

    def save_weights_evolution():                                       # OBSOLETE - using save_current_weights() instead
        print('shelving weights evolution')
        # with statement causes close() at the end, else must call close() explictly
        # 'c' opens for read and write, creating it if not existing
        # protocol = -1 uses the highest protocol (currently 2) which is binary,
        #  default protocol=0 is ascii and gives `ValueError: insecure string pickle` on loading
        with contextlib.closing(
                shelve.open(dataFileName+'_weights.shelve', 'c', protocol=-1)
                ) as data_dict:
            data_dict['Tmax'] = Tmax
            data_dict['errorLearning'] = errorLearning
            if errorLearning:
                data_dict['recurrentLearning'] = recurrentLearning
                data_dict['learnedWeights'] = sim.data[learnedWeightsProbe]
                data_dict['learnedInWeights'] = sim.data[learnedInWeightsProbe]
                data_dict['copycatLayer'] = copycatLayer
                #if recurrentLearning and copycatLayer:
                #    data_dict['copycatWeights'] = EtoEweights
                #    data_dict['copycatWeightsPert'] = EtoEweightsPert

    def save_current_weights(init,t):
        if not saveWeightsEvolution: return                             # not saving weights evolution as it takes up too much disk space
        if errorLearning:
            with contextlib.closing(
                    # 'c' opens for read/write, creating if non-existent
                    shelve.open(dataFileName+'_currentweights.shelve', 'c', protocol=-1)
                    ) as data_dict:
                if init:                                                # CAUTION: minor bug here, if weights are loaded from a file,
                                                                        #  then should save: sim.signals[ sim.model.sig[plasticConnEE]['weights'] ]
                    # data_dict in older file may have data, reassigned here
                    if plastDecoders:
                        data_dict['weights'] = np.array([sim.data[EtoE].weights])
                        data_dict['encoders'] = Eencoders
                        data_dict['reprRadius'] = ratorOut.radius
                        data_dict['gain'] = sim.data[ratorOut].gain
                    else:
                        data_dict['weights'] = np.array([sim.data[EtoE].weights])
                    data_dict['weightdt'] = weightdt
                    data_dict['Tmax'] = Tmax
                    # feedforward decoders or weights
                    data_dict['weightsIn'] = np.array([sim.data[InEtoE].weights])
                else:
                    if len(sim.data[learnedWeightsProbe]) > 0:
                        # since writeback=False by default in shelve.open(),
                        #  extend() / append() won't work directly,
                        #  so use a temp variable wts
                        #  see https://docs.python.org/2/library/shelve.html
                        wts = data_dict['weights']
                        wts = np.append(wts,sim.data[learnedWeightsProbe],axis=0)
                        data_dict['weights'] = wts
                        wts = data_dict['weightsIn']
                        wts = np.append(wts,sim.data[learnedInWeightsProbe],axis=0)
                        data_dict['weightsIn'] = wts
                        # flush the probe to save memory
                        del sim._probe_outputs[learnedWeightsProbe][:]
                        del sim._probe_outputs[learnedInWeightsProbe][:]

    _,_,_,_,realtimeold = os.times()
    def sim_run_flush(tFlush,nFlush):
        '''
            Run simulation for nFlush*tFlush seconds,
            Flush probes every tFlush of simulation time,
              (only flush those that don't have 'weights' in their label names)
        '''        
        weighttimeidxold = 0
        #doubledLearningRate = False
        for duration in [tFlush]*nFlush:
            _,_,_,_,realtime = os.times()
            print("Finished till",sim.time,'s, in',realtime-realtimeold,'s')
            sys.stdout.flush()
            # save weights if weightdt or more has passed since last save
            weighttimeidx = int(sim.time/weightdt)
            if weighttimeidx > weighttimeidxold:
                weighttimeidxold = weighttimeidx
                save_current_weights(False,sim.time)
            # flush probes
            for probe in sim.model.probes:
                # except weight probes (flushed in save_current_weights)
                # except error probe which is saved fully in ..._end.shelve
                if probe.label is not None:
                    if 'weights' in probe.label or 'error' in probe.label:
                        break
                del sim._probe_outputs[probe][:]
            ## if time > 1000s, double learning rate
            #if sim.time>1000. and not doubledLearningRate:
            #    changeLearningRate(4.)  # works only if excPESDecayRate = None
            #    doubledLearningRate = True
            # run simulation for tFlush duration
            sim.run(duration,progress_bar=False)

    ###
    ### run the simulation, with flushing for learning simulations ###
    ###
    if errorLearning:
        save_current_weights(True,0.)
        sim.run(Tnolearning)
        save_data('_start')
#         nFlush = int((Tmax-2*Tnolearning)/Tperiod)
#         sim_run_flush(Tperiod,nFlush)                       # last Tperiod remains (not flushed)
#         # turning learning off by modifying weight decay in some op-s is not needed
#         #  (haven't checked if it can be done in nengo_ocl like I did in nengo)
#         # I'm already setting error node to zero! If error was a spiking ensemble,
#         #  I'd have problems with spiking noise causing some 'learning', but with a node, it's fine!
#         #turn_off_learning()
#         save_current_weights(False,sim.time)
#         sim.run(Tnolearning)
#         save_current_weights(False,sim.time)
#         save_data('_end')
    else:
        sim.run(Tmax)
        save_data('')
    #save_weights_evolution()

    ###
    ### save the final learned exc weights; don't save if only testing older weights ###
    ###
    if errorLearning and not testLearned:
        with contextlib.closing(
                shelve.open(weightsSaveFileName, 'c', protocol=-1)
                ) as weights_dict:
            #weights_dict['learnedWeights'] = sim.data[plasticConnEE].weights       # this only saves the initial weights
            if sparseWeights:
                weights_dict['learnedWeights'] = sim.signals[ sim.model.sig[plasticConnEE]['weights'] ] * \
                                                    sim.signals[ sim.model.sig[plasticConnEE]['sparsity'] ]
                                                                                    # this is the signal updated by operator-s set by the learning rule
                weights_dict['learnedWeightsIn'] = sim.signals[ sim.model.sig[InEtoE]['weights'] ] *\
                                                    sim.signals[ sim.model.sig[InEtoE]['sparsity'] ]
                print(sim.signals[ sim.model.sig[plasticConnEE]['weights'] ] * \
                                                    sim.signals[ sim.model.sig[plasticConnEE]['sparsity'] ])
                                                                                    # this is the signal updated by operator-s set by the learning rule
            else:
                weights_dict['learnedWeights'] = sim.signals[ sim.model.sig[plasticConnEE]['weights'] ]
                                                                                    # this is the signal updated by operator-s set by the learning rule
                weights_dict['learnedWeightsIn'] = sim.signals[ sim.model.sig[InEtoE]['weights'] ]
                                                                                    # this is the signal updated by operator-s set by the learning rule
        print('saved end weights to',weightsSaveFileName)

    ###
    ### run the plotting sequence ###
    ###
    print('plotting data')
    myplot.plot_rec_nengo_all(dataFileName)
