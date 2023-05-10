import asyncio
import random
from typing import List, Dict
from collections import defaultdict
import sympy
from utils import *
import math
from setup import setup, gen
import socket
import json
from gui import NodeGUI
import time
import sys

sys.set_int_max_str_digits(0)

NUM_NODES = 5
MAL_NODES = 2
TIME_DELAY = 3
MAL_VAL = 10000
DATA_STORE = 1000000000000
random.seed(0)

class NodeBicorn:
    def __init__(self, id, gui, status):
        self.id = id
        self.rounds_received = defaultdict(set)
        self.current_round = 0
        self.received_messages_count = 0
        self.gui = gui
        self.time_taken_per_round = [-1]
        self.data_received_per_round = [-1]
        self.round_start_time = 0
        self.prev_output = 0
        self.status = status

        self.alpha = 0
        self.commit = 0
        self.x_curr = 0

    # Update the GUI when a new round is reached
    def update_gui(self):
        self.gui.update_node_info(self.id, self.current_round, self.x_curr, self.time_taken_per_round[-1], self.data_received_per_round[-1], self.status)

    def add_params(self, lamb, t, N, g, h, b):
        self.lamb = lamb
        self.t = t
        self.N = N
        self.g = g
        self.h = h
        self.b = b

    def _prepare(self):
        self.alpha = random.randint(1, self.b)
        self.commit = pow(self.g, self.alpha)
        
        if self.status == "mal":
            self.commit += random.randint(MAL_VAL, MAL_VAL*2)
        
        return

    def _commit(self):
        return self.commit

    def _reveal(self):
        return self.alpha

    def _pre_commit(self):
        return

    def _verify_commit(self, commit, g, alpha):
        term = pow(g, alpha)
        if commit == term:
            return True
        return False


    def _recover(self, commit, t):
        rec = pow(commit, pow(2,t))
        return rec

    def _finalize(self, alpha_commit_array, senders, g, h, t):
        final = 1
        
        for i in senders:
            
            if self._verify_commit(int(alpha_commit_array[i][1]), g, int(alpha_commit_array[i][0])):
                #print("At honest step")
                final *= pow(h, int(alpha_commit_array[i][0]))
            else:
                #print("At recover step")
                final *= self._recover(int(alpha_commit_array[i][1]), t)
            
    
        self.x_curr = final
        return final


    async def start(self):
        server = await asyncio.start_server(self.server_callback, 'localhost', 9001 + self.id)
        self.round_start_time = time.time()
        self.data_received_in_bytes = 0
        async with server:
            await asyncio.gather(
                asyncio.create_task(self.send_messages())
            )

    async def send_messages(self):
        await asyncio.sleep(1)  # Wait for all nodes to start

        while True:
            self._prepare()
            msg = {"sender": self.id, "round": self.current_round, "curr_alpha": str(self._reveal()), "curr_commit": str(self._commit())}
            sender_options = [i for i in range(self.lamb)]
            random.shuffle(sender_options)
            for i in sender_options:
                await self.send_message_to_socket(i, msg)

            await asyncio.sleep(0.1)  # Wait for the next round

    async def send_message_to_socket(self, target_id, message):
        try:
            reader, writer = await asyncio.open_connection(f'localhost', 9001 + target_id)
            writer.write(json.dumps(message).encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except ConnectionRefusedError:
            pass

    async def server_callback(self, reader, writer):
        data = await reader.read(DATA_STORE)
        self.data_received_in_bytes += len(data)
        msg = json.loads(data.decode())

        round_num = msg["round"]
        msg_sender = msg["sender"]
        curr_alpha = msg["curr_alpha"]
        curr_commit = msg["curr_commit"]
        self.received_messages_count += 1

        if round_num == self.current_round:
            self.rounds_received[round_num].add((msg_sender+1, curr_alpha, curr_commit))
            if len(self.rounds_received[round_num]) == self.lamb:
                senders = [x[0] for x in self.rounds_received[round_num]]
                alpha_commit_array = {x[0]: (x[1],x[2]) for x in self.rounds_received[round_num]}
                x_next = self._finalize(alpha_commit_array, senders, self.g, self.h, self.t)
            
                self.current_round += 1
                #print(f"Reached round {self.current_round} for node {self.id} -> Message = {self.x_curr}")
                current_time = time.time()
                self.time_taken_per_round.append(current_time - self.round_start_time)
                self.data_received_per_round.append(self.data_received_in_bytes)
                self.data_received_in_bytes = 0
                self.round_start_time = current_time
                self.update_gui()

        writer.close()
        await writer.wait_closed()

class NodeStrobe:
    def __init__(self, id, gui, status):
        self.id = id
        self.rounds_received = defaultdict(set)
        self.current_round = 0
        self.received_messages_count = 0
        self.gui = gui
        self.time_taken_per_round = [-1]
        self.data_received_per_round = [-1]
        self.round_start_time = 0
        self.prev_output = 0
        self.status = status

    # Update the GUI when a new round is reached
    def update_gui(self):
        self.gui.update_node_info(self.id, self.current_round, self.x_curr, self.time_taken_per_round[-1], self.data_received_per_round[-1], self.status)

    def add_params(self, n, phi, N, s, a_coeff, x_0, t):
        self.N = N
        self.s = s
        self.n = n
        v = sympy.mod_inverse(math.factorial(n)*s, phi//4)
        self.sk = v + sum([a_coeff[j-1]*(self.id+1)**j for j in range(1, len(a_coeff)+1)])
        self.x_curr = x_0
        self.t = t

    def _eval(self):
        x_next_i = pow(self.x_curr, self.sk, self.N)
        self.prev_output = x_next_i
        if self.status == "mal" and self.current_round > 0:
            return MAL_VAL
        return x_next_i

    def _verify_share(self, x_next_i, x_curr_i):
        x_curr_i %= self.N
        if x_curr_i == pow(x_next_i, self.s, self.N):
            return True
        return False

    def _combine(self, x_next_array, selected_indices):
        n_fact = math.factorial(self.n)
        n_fact_times_L_0 = {i: lagrange_basis_polynomial(i, 0, selected_indices, n_fact) for i in range(1, self.n+1)}
        x_next = 1
        for i in selected_indices:
            x_next *= pow(x_next_array[i], n_fact_times_L_0[i], self.N)
        x_next %= self.N
        return x_next

    def _verify(self, x_next, x_curr):
        x_curr %= self.N
        if x_curr == pow(x_next, self.s, self.N):
            return True
        return False

    async def start(self):
        server = await asyncio.start_server(self.server_callback, 'localhost', 8000 + self.id)
        self.round_start_time = time.time()
        self.data_received_in_bytes = 0
        async with server:
            await asyncio.gather(
                asyncio.create_task(self.send_messages())
            )

    async def send_messages(self):
        await asyncio.sleep(1)  # Wait for all nodes to start

        while True:
            msg = {"sender": self.id, "round": self.current_round, "prev_output": self.prev_output, "x_part": self._eval()}
            options = [i for i in range(self.n)] # if i != self.id]
            random.shuffle(options)
            for i in options:
                # if i != self.id:
                await self.send_message_to_socket(i, msg)
                # await asyncio.sleep(random.uniform(0.1, 0.5))  # Add random small delay

            await asyncio.sleep(0.1)  # Wait for the next round

    async def send_message_to_socket(self, target_id, message):
        try:
            reader, writer = await asyncio.open_connection(f'localhost', 8000 + target_id)
            writer.write(json.dumps(message).encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except ConnectionRefusedError:
            pass

    async def server_callback(self, reader, writer):
        data = await reader.read(100)
        self.data_received_in_bytes += len(data)
        msg = json.loads(data.decode())
        # if self.id == 2:
        #     print(f"Node {self.id} received message {msg}")

        round_num = msg["round"]
        msg_sender = msg["sender"]
        x_part = msg["x_part"]
        prev_out = msg["prev_output"]
        self.received_messages_count += 1

        if round_num == self.current_round:
            if self.current_round == 0 or self._verify_share(x_part, prev_out):
                self.rounds_received[round_num].add((msg_sender+1, x_part))

                if len(self.rounds_received[round_num]) >= self.t:
                    senders = [x[0] for x in self.rounds_received[round_num]]
                    x_next_array = {x[0]: x[1] for x in self.rounds_received[round_num]}
                    x_next = self._combine(x_next_array, senders)
                    if self._verify(x_next, self.x_curr):
                        # print(f"Node {self.id} verified the share")
                        self.x_curr = x_next
                    # else:
                        # print(f"Node {self.id} failed to verify the share")
                
                    self.current_round += 1
                    print(f"Reached round {self.current_round} for node {self.id} -> Message = {self.x_curr}")
                    current_time = time.time()
                    self.time_taken_per_round.append(current_time - self.round_start_time)
                    self.data_received_per_round.append(self.data_received_in_bytes)
                    self.data_received_in_bytes = 0
                    self.round_start_time = current_time
                    self.update_gui()

        writer.close()
        await writer.wait_closed()


async def main(n, m, t, alg):

    if alg == "strobe":
        N, phi, s, a_coeff = setup(n, t, alg)
        x_0 = gen(math.factorial(n), N)
        gui = NodeGUI(n, alg)
        nodes = [NodeStrobe(i, gui, "honest") for i in range(n-m)] + [NodeStrobe(i, gui, "mal") for i in range(n-m, n)]
        for node in nodes:
            node.add_params(n, phi, N, s, a_coeff, x_0, t)

        tasks = [node.start() for node in nodes] + [update_gui_periodically(gui)]

        await asyncio.gather(*tasks)

    elif alg == "bicorn":
        N, g, h, b = setup(n, t, alg)
        gui = NodeGUI(n, alg)
        nodes = [NodeBicorn(i, gui, "honest") for i in range(n-m)] + [NodeBicorn(i, gui, "mal") for i in range(n-m, n)]
        for node in nodes:
            node.add_params(n, t, N, g, h, b)

        tasks = [node.start() for node in nodes] + [update_gui_periodically(gui)]

        await asyncio.gather(*tasks)

    else:
        print("Algorithm not implemented")
        return

if __name__ == "__main__":
    args = sys.argv[1:]
    asyncio.run(main(int(args[0]), int(args[1]), int(args[2]), args[3]))
