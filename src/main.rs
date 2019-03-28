/*
    A Rust implementation of Riccardo Poli's TinyGP:
        https://cswww.essex.ac.uk/staff/rpoli/TinyGP/
    closely based on his Java version from here:
        https://cswww.essex.ac.uk/staff/rpoli/TinyGP/tiny_gp.java

    March 2019 by John Green (john@joanju.com)

    This closely follows Riccardo's java code, so that this rust version
    can easily be compared with the java version. I intend to do some variation
    on this, but as a separate fork.
*/


// Allow names to be the same as they were in Riccardo's java version
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]


use rand::prelude::*;
use std::error::Error;
use std::io::prelude::*;
use std::str::FromStr;


const ADD: u8 = 110;
const SUB: u8 = 111; 
const MUL: u8 = 112; 
const DIV: u8 = 113;
const FSET_START: u8 = ADD; 
const FSET_END: u8 = DIV;
const MAX_LEN: usize = 10_000;
const POPSIZE: usize = 100_000;
const DEPTH: usize = 5;
const GENERATIONS: usize = 100;
const TSIZE: usize = 2;
const PMUT_PER_NODE: f64 = 0.05;
const CROSSOVER_PROB: f64 = 0.9;


fn main() {
    let mut filename = "problem.dat";
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut seed: i64 = -1;
    if args.len() == 2 {
        match u32::from_str(&args[0]) {
             Err(_) => {
                println!("Error parsing args: First arg must be a non-negative integer");
                return;
            },
            Ok(i) => {seed = i as i64}
        }
        filename = &args[1];
    }
    else if args.len() == 1 {
        filename = &args[0];
    }
    let mut gp = tiny_gp::new(seed, filename);
    if let Err(err) = gp.setup() {
        println!("{}", err);
        return;
    }
    gp.evolve();
}


struct tiny_gp {
    favgpop: f64,
    fbestpop: f64,
    filename: String,
    fitness: Vec<f64>,
    maxrandom: f64,
    minrandom: f64,
    PC: usize,
    pop: Vec<Vec<u8>>,
    rd: StdRng,
    seed: i64,
    targets: Vec<Vec<f64>>,
    x: [f64; FSET_START as usize],
    varnumber: u8, fitnesscases: u8, randomnumber: u8,
}


impl tiny_gp {

    pub fn new(seed: i64, filename: &str) -> tiny_gp {
        tiny_gp {
            favgpop: 0.0,
            fbestpop: 0.0,
            filename: filename.to_string(),
            fitness: vec![0.0; POPSIZE],
            maxrandom: 0.0,
            minrandom: 0.0,
            PC: 0,
            pop: Vec::new(),
            rd: StdRng::from_entropy(),
            seed: seed,
            targets: Vec::new(),
            x: [0.0; FSET_START as usize],
            varnumber: 0, fitnesscases: 0, randomnumber: 0,
        }
    }

    fn setup(&mut self) -> Result<(), Box<Error>> {
        if self.seed >= 0 {
            self.rd = SeedableRng::seed_from_u64(self.seed as u64);
        }
        self.setup_fitness()?;
        for i in 0..FSET_START as usize {
            self.x[i] =
                (self.maxrandom - self.minrandom) * self.rd.gen::<f64>()
                + self.minrandom;
        }
        self.pop = self.create_random_pop(POPSIZE, DEPTH);
        Ok(())
    }

    // The interpreter
    fn run(&mut self, program: &[u8]) -> f64 {
        let primitive = program[self.PC];
        self.PC += 1;
        if primitive < FSET_START {
            return self.x[primitive as usize];
        }
        match primitive {
            ADD => self.run(program) + self.run(program),
            SUB => self.run(program) - self.run(program),
            MUL => self.run(program) * self.run(program),
            DIV => {
                let num = self.run(program);
                let den = self.run(program);
                if den.abs() <= 0.001 {
                    return num;
                }
                num / den
            }
            _ => 0.0  // should never get here
        }
    }

    fn traverse(&self, buffer: &[u8], buffercount: usize) -> usize {
        if buffer[buffercount] < FSET_START {
            return buffercount + 1;
        }
        match buffer[buffercount] {
            ADD | SUB | MUL | DIV =>
                self.traverse(buffer, self.traverse(buffer, buffercount + 1)),
            _ => 0 // should never get here
        }
    }

    fn setup_fitness(&mut self) -> Result<(), Box<Error>> {
        let file = std::fs::File::open(&self.filename)?;
        let reader = std::io::BufReader::new(file);
        let mut line_iter = reader.lines();
        let errmsg = "Invalid data file";
        let mut line = line_iter.next().ok_or(errmsg)??;
        let mut token_iter = line.split_whitespace();
        self.varnumber = u8::from_str(token_iter.next().ok_or(errmsg)?)?;
        self.randomnumber = u8::from_str(token_iter.next().ok_or(errmsg)?)?;
        self.minrandom = f64::from_str(token_iter.next().ok_or(errmsg)?)?;
        self.maxrandom = f64::from_str(token_iter.next().ok_or(errmsg)?)?;
        self.fitnesscases = u8::from_str(token_iter.next().ok_or(errmsg)?)?;
        if self.varnumber + self.randomnumber >= FSET_START {
            return Err("too many variables and constants".into());
        }
        self.targets = Vec::new();
        for _ in 0..self.fitnesscases {
            line = line_iter.next().ok_or(errmsg)??;
            let mut token_iter = line.split_whitespace();
            let mut v = Vec::new();
            for _ in 0..=self.varnumber {
                let val = f64::from_str(token_iter.next().ok_or(errmsg)?)?;
                v.push(val);
            }
            self.targets.push(v);
        }
        Ok(())
    }

    fn fitness_function(&mut self, Prog: &Vec<u8>) -> f64 {
        let mut fit: f64 = 0.0;
        self.traverse(Prog, 0);
        let varnum = self.varnumber as usize;
        for i in 0..self.fitnesscases as usize {
            self.x[..varnum].clone_from_slice(&self.targets[i][..varnum]);
            self.PC = 0;
            let result = self.run(Prog);
            fit += (result - self.targets[i][varnum]).abs();
        }
        -fit
    }

    fn grow(&mut self, buffer: &mut Vec<u8>, pos: usize, max: usize, depth: usize) -> isize {
        if pos >= max {
            return -1;
        }
        let mut prim: u8 = if pos == 0 { 1 } else { self.rd.gen_range(0, 2) };
        if prim == 0 || depth == 0 {
            prim = self.rd.gen_range(0, self.varnumber + self.randomnumber);
            buffer[pos] = prim;
            pos as isize + 1
        }
        else  {
            prim = self.rd.gen_range(0, FSET_END - FSET_START + 1) + FSET_START;
            match prim {
                ADD | SUB | MUL | DIV => {},
                _ => { return 0; }  // should never get here
            }
            buffer[pos] = prim;
            let one_child = self.grow(buffer, pos + 1, max, depth - 1);
            if one_child < 0 {
                return -1;
            }
            self.grow(buffer, one_child as usize, max, depth - 1)
        }
    }

    fn print_indiv(&self, buffer: &[u8], buffercounter: usize) -> usize {
        let mut a1: usize = 0;
        if buffer[buffercounter] < FSET_START {
            if buffer[buffercounter] < self.varnumber {
                print!("X{} ", buffer[buffercounter] + 1);
            } else {
                print!("{}", &self.x[buffer[buffercounter] as usize]);
            }
            return buffercounter + 1;
        }
        match buffer[buffercounter] {
            ADD => {
                print!("(");
                a1 = self.print_indiv(buffer, buffercounter + 1); 
                print!(" + ");
            }
            SUB => {
                print!("(");
                a1 = self.print_indiv(buffer, buffercounter + 1);
                print!(" - "); 
            }
            MUL => {
                print!("(");
                a1 = self.print_indiv(buffer, buffercounter + 1);
                print!(" * ");
            }
            DIV => {
                print!("(");
                a1 = self.print_indiv(buffer, buffercounter + 1);
                print!(" / ");
            }
            _ => {}
        }
        let a2 = self.print_indiv(buffer, a1);
        print!(")");
        a2
    }

    fn create_random_indiv(&mut self, depth: usize) -> Vec<u8> {
        let mut ind: Vec<u8> = vec![0; MAX_LEN];
        let mut len = self.grow(&mut ind, 0, MAX_LEN, depth);
        while len < 0 {
            len = self.grow(&mut ind, 0, MAX_LEN, depth);
        }
        ind
    }

    fn create_random_pop(&mut self, n: usize, depth: usize) -> Vec<Vec<u8>> {
        let mut pop: Vec<Vec<u8>> = Vec::new();
        for i in 0..n {
            pop.push(self.create_random_indiv(depth));
            self.fitness[i] = self.fitness_function(&pop[i]);
        }
        pop
    }

    fn stats(&mut self, gen: usize) {
        let mut best = self.rd.gen_range(0, POPSIZE);
        let mut node_count: usize = 0;
        self.fbestpop = self.fitness[best];
        self.favgpop = 0.0;
        for i in 0..POPSIZE {
            node_count += self.traverse(&self.pop[i], 0);
            self.favgpop += self.fitness[i];
            if self.fitness[i] > self.fbestpop {
                best = i;
                self.fbestpop = self.fitness[i];
            }
        }
        let fpopsize = POPSIZE as f64;
        let avg_len = (node_count as f64) / fpopsize;
        self.favgpop /= fpopsize;
        print!("\
            Generation={} \
            Avg Fitness={} \
            Best Fitness={} \
            Avg Size={} \n\
            Best Individual: ",
            gen, -self.favgpop, -self.fbestpop, avg_len
            );
        let buffer = &self.pop[best];
        self.print_indiv(buffer, 0);
        println!();
    }

    fn tournament(&mut self, tsize: usize) -> usize {
        let mut best = self.rd.gen_range(0, POPSIZE);
        let mut fbest: f64 = -1.0e34;
        for _ in 0..tsize {
            let competitor = self.rd.gen_range(0, POPSIZE);
            if self.fitness[competitor] > fbest {
                fbest = self.fitness[competitor];
                best = competitor;
            }
        }
        best
    }

    fn negative_tournament(&mut self, tsize: usize) -> usize {
        let mut worst = self.rd.gen_range(0, POPSIZE);
        let mut fworst: f64 = 1e34;
        for _ in 0..tsize {
            let competitor = self.rd.gen_range(0, POPSIZE);
            if self.fitness[competitor] < fworst {
                fworst = self.fitness[competitor];
                worst = competitor;
            }
        }
        worst
    }

    fn crossover(&mut self, parent1Index: usize, parent2Index: usize) -> Vec<u8> {
        let parent1 = &self.pop[parent1Index];
        let parent2 = &self.pop[parent2Index];
        let len1 = self.traverse(&parent1, 0);
        let len2 = self.traverse(&parent2, 0);
        let xo1start = self.rd.gen_range(0, len1);
        let xo1end = self.traverse(&parent1, xo1start);
        let xo2start = self.rd.gen_range(0, len2);
        let xo2end = self.traverse(&parent2, xo2start);
        let mut offspring = (&parent1[0..xo1start]).to_vec();
        offspring.extend((&parent2[xo2start..xo2end]).iter());
        offspring.extend((&parent1[xo1end..len1]).iter());
        offspring
    }

    fn mutation(&mut self, parentIndex: usize, pmut: f64) -> Vec<u8> {
        let parent = &self.pop[parentIndex];
        let len = self.traverse(&parent, 0);
        let mut parentcopy = parent.clone();
        for i in 0..len {
            if self.rd.gen::<f64>() < pmut {
                let mutsite = i;
                if parentcopy[mutsite] < FSET_START {
                    parentcopy[mutsite] =
                        self.rd.gen_range(0, self.varnumber + self.randomnumber);
                }
                else {
                    match parentcopy[mutsite] {
                        ADD | SUB | MUL | DIV => {
                            parentcopy[mutsite] =
                                self.rd.gen_range(0, FSET_END - FSET_START + 1) + FSET_START;
                        },
                        _ => {}
                    }
                }
            }
        }
        parentcopy
    }

    fn print_parms(&self) {
        println!("\
            -- TINY GP (Rust implementation) -- \n\
            SEED={} \n\
            MAX_LEN={} \n\
            POPSIZE={} \n\
            DEPTH={} \n\
            CROSSOVER_PROB={} \n\
            PMUT_PER_NODE={} \n\
            MIN_RANDOM={} \n\
            MAX_RANDOM={} \n\
            GENERATIONS={} \n\
            TSIZE={} \n\
            ----------------------------------",
            self.seed, MAX_LEN, POPSIZE, DEPTH, CROSSOVER_PROB, PMUT_PER_NODE,
            self.minrandom, self.maxrandom, GENERATIONS, TSIZE
            );
    }

    pub fn evolve(&mut self) {
        let mut offspring;
        let mut parent1;
        let mut parent2;
        let mut parent;
        let mut newfit;
        let mut newind;
        self.print_parms();
        self.stats(0);
        for gen in 1..GENERATIONS {
            if self.fbestpop > -1e-5 {
                println!("PROBLEM SOLVED");
                std::process::exit(0);
            }
            for _ in 0..POPSIZE {
                if self.rd.gen::<f64>() < CROSSOVER_PROB {
                    parent1 = self.tournament(TSIZE);
                    parent2 = self.tournament(TSIZE);
                    newind = self.crossover(parent1, parent2);
                } else {
                    parent = self.tournament(TSIZE);
                    newind = self.mutation(parent, PMUT_PER_NODE);
                }
                newfit = self.fitness_function(&newind);
                offspring = self.negative_tournament(TSIZE);
                self.pop[offspring] = newind;
                self.fitness[offspring] = newfit;
            }
            self.stats(gen);
        }
        println!("PROBLEM *NOT* SOLVED");
        std::process::exit(1);
    }

}
