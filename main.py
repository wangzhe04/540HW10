# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

num_of_episodes = 100000

for episode in range(0, num_of_episodes):
    # Reset the enviroment
    state = enviroment.reset()

    # Initialize variables
    reward = 0
    terminated = False

    while not terminated:
        # Take learned path or explore new actions based on the epsilon
        if random.uniform(0, 1) < epsilon:
            action = enviroment.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Take action
        next_state, reward, terminated, info = enviroment.step(action)

        # Recalculate
        q_value = q_table[state, action]
        max_value = np.max(q_table[next_state])
        new_q_value = (1 – alpha) * q_value + alpha * (reward + gamma * max_value)

        # Update Q-table
        q_table[state, action] = new_q_value
        state = next_state

    if (episode + 1) % 100 == 0:
        clear_output(wait=True)
        print("Episode: {}".format(episode + 1))
        enviroment.render()

print("**********************************")
print("Training is done!\n")
print("**********************************")
