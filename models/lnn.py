import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import stax, optimizers
from jax.experimental.ode import odeint
from functools import partial


# unconstrained equation of motion
def unconstrained_eom(model, state, t=None):
    q, q_t = jnp.split(state, 2)
    return model(q, q_t)


# lagrangian equation of motion
def lagrangian_eom(lagrangian, state, t=None):
    q, q_t = jnp.split(state, 2)
    # Note: the following line assumes q is an angle. Delete it for problems other than double pendulum.
    q = q % (2 * jnp.pi)
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
            @ (jax.grad(lagrangian, 0)(q, q_t)
               - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
    dt = 1e-1
    return dt * jnp.concatenate([q_t, q_tt])


def raw_lagrangian_eom(lagrangian, state, t=None):
    q, q_t = jnp.split(state, 2)
    q = q % (2 * jnp.pi)
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
            @ (jax.grad(lagrangian, 0)(q, q_t)
               - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
    return jnp.concatenate([q_t, q_tt])


def lagrangian_eom_rk4(lagrangian, state, n_updates, Dt=1e-1, t=None):
    @jax.jit
    def cur_fnc(state):
        q, q_t = jnp.split(state, 2)
        q = q % (2 * jnp.pi)
        q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
                @ (jax.grad(lagrangian, 0)(q, q_t)
                   - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
        return jnp.concatenate([q_t, q_tt])

    @jax.jit
    def get_update(update):
        dt = Dt / n_updates
        cstate = state + update
        k1 = dt * cur_fnc(cstate)
        k2 = dt * cur_fnc(cstate + k1 / 2)
        k3 = dt * cur_fnc(cstate + k2 / 2)
        k4 = dt * cur_fnc(cstate + k3)
        return update + 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    update = 0
    for _ in range(n_updates):
        update = get_update(update)
    return update


def solve_dynamics(dynamics_fn, initial_state, is_lagrangian=True, **kwargs):
    eom = lagrangian_eom if is_lagrangian else unconstrained_eom

    # We currently run odeint on CPUs only, because its cost is dominated by
    # control flow, which is slow on GPUs.
    @partial(jax.jit, backend='cpu')
    def f(initial_state):
        return odeint(partial(eom, dynamics_fn), initial_state, **kwargs)

    return f(initial_state)


def custom_init(init_params, seed=0):
    """Do an optimized LNN initialization for a simple uniform-width MLP"""
    import numpy as np
    new_params = []
    rng = jax.random.PRNGKey(seed)
    i = 0
    number_layers = len([0 for l1 in init_params if len(l1) != 0])
    for l1 in init_params:
        if (len(l1)) == 0: new_params.append(()); continue
        new_l1 = []
        for l2 in l1:
            if len(l2.shape) == 1:
                # Zero init biases
                new_l1.append(jnp.zeros_like(l2))
            else:
                n = max(l2.shape)
                first = int(i == 0)
                last = int(i == number_layers - 1)
                mid = int((i != 0) * (i != number_layers - 1))
                mid *= i

                std = 1.0 / np.sqrt(n)
                std *= 2.2 * first + 0.58 * mid + n * last

                if std == 0:
                    raise NotImplementedError("Wrong dimensions for MLP")

                new_l1.append(jax.random.normal(rng, l2.shape) * std)
                rng += 1
                i += 1

        new_params.append(new_l1)

    return new_params


def mlp(args):
    return stax.serial(
        stax.Dense(args.hidden_dim),
        stax.Softplus,
        stax.Dense(args.hidden_dim),
        stax.Softplus,
        stax.Dense(args.output_dim),
    )


def pixel_encoder(args):
    return stax.serial(
        stax.Dense(args.ae_hidden_dim),
        stax.Softplus,
        stax.Dense(args.ae_latent_dim),
    )


def pixel_decoder(args):
    return stax.serial(
        stax.Dense(args.ae_hidden_dim),
        stax.Softplus,
        stax.Dense(args.ae_input_dim),
    )


def wrap_coords(state):
    # wrap generalized coordinates to [-pi, pi]
    return jnp.concatenate([(state[:2] + jnp.pi) % (2 * jnp.pi) - jnp.pi, state[2:]])


def rk4_step(f, x, t, h):
    # one step of Runge-Kutta integration
    k1 = h * f(x, t)
    k2 = h * f(x + k1/2, t + h/2)
    k3 = h * f(x + k2/2, t + h/2)
    k4 = h * f(x + k3, t + h)
    return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)


def radial2cartesian(t1, t2, l1, l2):
    # Convert from radial to Cartesian coordinates.
    x1 = l1 * jnp.sin(t1)
    y1 = -l1 * jnp.cos(t1)
    x2 = x1 + l2 * jnp.sin(t2)
    y2 = y1 - l2 * jnp.cos(t2)
    return x1, y1, x2, y2


def write_to(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_from(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# replace the lagrangian with a parameteric model
def learned_dynamics(params):
    def dynamics(q, q_t):
        assert q.shape == (2,)
        state = wrap_coords(jnp.concatenate([q, q_t]))
        return jnp.squeeze(nn_forward_fn(params, state), axis=-1)
    return dynamics


@jax.jit
def gln_loss(params, batch, time_step=None):
    state, targets = batch
    preds = jax.vmap(partial(lagrangian_eom, learned_dynamics(params)))(state)
    return jnp.mean((preds - targets) ** 2)


@jax.jit
def baseline_loss(params, batch, time_step=None):
    state, targets = batch
    preds = jax.vmap(partial(unconstrained_eom, learned_dynamics(params)))(state)
    return jnp.mean((preds - targets) ** 2)


def train(args, model, data):
    global opt_update, get_params, nn_forward_fn
    (nn_forward_fn, init_params) = model
    data = {k: jax.device_put(v) if type(v) is jnp.ndarray else v for k, v in data.items()}
    time.sleep(2)

    # choose our loss function
    if args.model == 'gln':
        loss = gln_loss
    elif args.model == 'baseline_nn':
        loss = baseline_loss
    else:
        raise ValueError

    @jax.jit
    def update_derivative(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, jax.grad(loss)(params, batch, None), opt_state)

    # make an optimizer
    opt_init, opt_update, get_params = optimizers.adam(
        lambda t: jnp.select([t < args.batch_size*(args.num_batches//3),
                              t < args.batch_size*(2*args.num_batches//3),
                              t > args.batch_size*(2*args.num_batches//3)],
                             [args.learn_rate, args.learn_rate/10, args.learn_rate/100]))
    opt_state = opt_init(init_params)

    train_losses, test_losses = [], []
    for iteration in range(args.batch_size*args.num_batches + 1):
        if iteration % args.batch_size == 0:
            params = get_params(opt_state)
            train_loss = loss(params, (data['x'], data['dx']))
            train_losses.append(train_loss)
            test_loss = loss(params, (data['test_x'], data['test_dx']))
            test_losses.append(test_loss)
            if iteration % (args.batch_size*args.test_every) == 0:
                print(f"iteration={iteration}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")
        opt_state = update_derivative(iteration, opt_state, (data['x'], data['dx']))

    params = get_params(opt_state)
    return params, train_losses, test_losses


@jax.jit
def kinetic_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8):
    (t1, t2), (w1, w2) = q, q_dot

    T1 = 0.5 * m1 * (l1 * w1) ** 2
    T2 = 0.5 * m2 * ((l1 * w1) ** 2 + (l2 * w2) ** 2 + 2 * l1 * l2 * w1 * w2 * jnp.cos(t1 - t2))
    T = T1 + T2
    return T


@jax.jit
def potential_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8):
    (t1, t2), (w1, w2) = q, q_dot

    y1 = -l1 * jnp.cos(t1)
    y2 = y1 - l2 * jnp.cos(t2)
    V = m1 * g * y1 + m2 * g * y2
    return V


# Double pendulum lagrangian
@jax.jit
def lagrangian_fn(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8):
    (t1, t2), (w1, w2) = q, q_dot

    T = kinetic_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8)
    V = potential_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8)
    return T - V


# Double pendulum lagrangian
@jax.jit
def hamiltonian_fn(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8):
    (t1, t2), (w1, w2) = q, q_dot

    T = kinetic_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8)
    V = potential_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8)
    return T + V


# Double pendulum dynamics via analytical forces taken from Diego's blog
@jax.jit
def analytical_fn(state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
    t1, t2, w1, w2 = state
    a1 = (l2 / l1) * (m2 / (m1 + m2)) * jnp.cos(t1 - t2)
    a2 = (l1 / l2) * jnp.cos(t1 - t2)
    f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2 ** 2) * jnp.sin(t1 - t2) - (g / l1) * jnp.sin(t1)
    f2 = (l1 / l2) * (w1 ** 2) * jnp.sin(t1 - t2) - (g / l2) * jnp.sin(t2)
    g1 = (f1 - a1 * f2) / (1 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1 - a1 * a2)
    return jnp.stack([w1, w2, g1, g2])


@partial(jax.jit, backend='cpu')
def get_trajectory(y0, times, use_lagrangian=False, **kwargs):
    # frames = int(fps*(t_span[1]-t_span[0]))
    # times = jnp.linspace(t_span[0], t_span[1], frames)
    # y0 = np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32)
    if use_lagrangian:
        y = solve_dynamics(lagrangian_fn, y0, t=times, is_lagrangian=True, rtol=1e-10, atol=1e-10, **kwargs)
    else:
        y = odeint(analytical_fn, y0, t=times, rtol=1e-10, atol=1e-10, **kwargs)
    return y


@partial(jax.jit, backend='cpu')
def get_trajectory_lagrangian(y0, times, **kwargs):
    return solve_dynamics(lagrangian_fn, y0, t=times, is_lagrangian=True, rtol=1e-10, atol=1e-10, **kwargs)


@partial(jax.jit, backend='cpu')
def get_trajectory_analytic(y0, times, **kwargs):
    return odeint(analytical_fn, y0, t=times, rtol=1e-10, atol=1e-10, **kwargs)


def get_dataset(seed=0, samples=1, t_span=[0, 2000], fps=1, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)

    frames = int(fps * (t_span[1] - t_span[0]))
    times = np.linspace(t_span[0], t_span[1], frames)
    y0 = np.array([3 * np.pi / 7, 3 * np.pi / 4, 0, 0], dtype=np.float32)

    xs, dxs = [], []
    vfnc = jax.jit(jax.vmap(analytical_fn))
    for s in range(samples):
        x = get_trajectory(y0, times, **kwargs)
        dx = vfnc(x)
        xs.append(x)
        dxs.append(dx)

    data['x'] = jax.vmap(wrap_coords)(jnp.concatenate(xs))
    data['dx'] = jnp.concatenate(dxs)
    data['t'] = times

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx', 't']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data


class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


if __name__ == "__main__":
    args = ObjectView({
        'input_dim': 4,
        'hidden_dim': 128,
        'output_dim': 1,
        'dataset_size': 3000,
        'learn_rate': 1e-3,
        'batch_size': 100,
        'test_every': 10,
        'num_batches': 500,
        'name': 'dblpend',
        'model': 'gln',
        'verbose': True,
        'seed': 1,
        'save_dir': '.'
    })

    rng = jax.random.PRNGKey(args.seed)
    init_random_params, nn_forward_fn = mlp(args)
    _, init_params = init_random_params(rng, (-1, 4))
    model = (nn_forward_fn, init_params)
    data = get_dataset(t_span=[0, args.dataset_size], fps=1, samples=1)

    result = train(args, model, data)
