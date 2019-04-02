/*
    WeeGP

    Variations on Riccardo Poli's TinyGP:
        https://cswww.essex.ac.uk/staff/rpoli/TinyGP/
        https://cswww.essex.ac.uk/staff/rpoli/TinyGP/tiny_gp.java

    March 2019 by John Green (john@joanju.com)

*/


use rand::prelude::*;
use std::error::Error;
use std::io::prelude::*;
use std::str::FromStr;
use std::time::Instant;

const ADD: u8 = 200;
const SUB: u8 = 201; 
const MUL: u8 = 202; 
const DIV: u8 = 203;
const FSET_START: u8 = ADD; 
const FSET_END: u8 = DIV;
const MAX_LEN: usize = 40;
const POPSIZE: usize = 1_000_000;
const INITIAL_EXPRESSION_DEPTH: usize = 5;
const GENERATIONS: usize = 2000;
const TOURNAMENT_SIZE: usize = 2;
const PMUT_PER_NODE: f64 = 0.05;
const CROSSOVER_PROB: f64 = 0.9;
const PRINT_FREQUENCY: usize = 2;
// For tournaments, if two competitors fitness are within this percent of
// each other, give preference to the smaller competitor.
const CONSIDERED_CLOSE: f64 = 0.01;


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
            Ok(value) => {seed = i64::from(value)}
        }
        filename = &args[1];
    }
    else if args.len() == 1 {
        filename = &args[0];
    }
    let mut gp = WeeGP::new(seed, filename);
    if let Err(err) = gp.setup() {
        println!("{}", err);
        return;
    }
    gp.evolve();
}


struct WeeGP {
    favgpop: f64,
    fbestpop: f64,
    filename: String,
    fitness: Vec<f64>,
    fitnesscases: u8,
    pc: usize,
    pop: Vec<Vec<u8>>,
    rd: StdRng,
    seed: i64,
    starttime: std::time::Instant,
    targets: Vec<Vec<f64>>,
    x: Vec<f64>,
    x_names: Vec<String>,
    varnumber: u8,
}


impl WeeGP {

    pub fn new(seed: i64, filename: &str) -> WeeGP {
        WeeGP {
            favgpop: 0.0,
            fbestpop: 0.0,
            filename: filename.to_string(),
            fitness: vec![0.0; POPSIZE],
            fitnesscases: 0,
            pc: 0,
            pop: Vec::new(),
            rd: StdRng::from_entropy(),
            seed: seed,
            starttime: Instant::now(),
            targets: Vec::new(),
            x: Vec::new(),
            x_names: Vec::new(),
            varnumber: 0,
        }
    }

    fn setup(&mut self) -> Result<(), Box<Error>> {
        if self.seed >= 0 {
            self.rd = SeedableRng::seed_from_u64(self.seed as u64);
        }
        self.setup_fitness()?;  // Loads the parameters
        self.setup_x();  // Uses parameter values to configure the x vector
        if self.x.len() >= FSET_START as usize {
            return Err("Too many independent variables".into());
        }
        self.pop = self.create_random_pop(POPSIZE, INITIAL_EXPRESSION_DEPTH);
        Ok(())
    }

    fn setup_fitness(&mut self) -> Result<(), Box<Error>> {
        let file = std::fs::File::open(&self.filename)?;
        let reader = std::io::BufReader::new(file);
        let mut line_iter = reader.lines();
        let errmsg = "Invalid data file";
        let mut line = line_iter.next().ok_or(errmsg)??;
        let mut token_iter = line.split_whitespace();
        self.varnumber = u8::from_str(token_iter.next().ok_or(errmsg)?)?;
        token_iter.next().ok_or(errmsg)?; // tinyGP 'randomnumber', not used here
        token_iter.next().ok_or(errmsg)?; // tinyGP 'minrandom', not used here
        token_iter.next().ok_or(errmsg)?; // tinyGP 'maxrandom', not used here
        self.fitnesscases = u8::from_str(token_iter.next().ok_or(errmsg)?)?;
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

    fn setup_x(&mut self) {
        // Add one slot for each independent variable from our input
        for n in 0..self.varnumber as usize {
            self.add_x(0.0, &format!("X{}", n));
        }
        // Now add some f64 constants
        self.add_x(-1.0, "-1");
        for n in 1..=10 {
            self.add_x(f64::from(n), &format!("{}", n));
        }
        self.add_x(100_f64, "100");
        self.add_x(1000_f64, "1000");
        self.add_x(std::f64::consts::PI, "pi");
        self.add_x(std::f64::consts::E, "e");
    }

    fn add_x(&mut self, value: f64, name: &str) {
            self.x.push(value);
            self.x_names.push(name.to_string());
    }

    // The interpreter
    fn run(&mut self, program: &[u8]) -> f64 {
        let primitive = program[self.pc];
        self.pc += 1;
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
            _ => panic!("Unexpected token during Run")
        }
    }

    fn expression_length(&self, buffer: &[u8], buffercount: usize) -> usize {
        if buffer[buffercount] < FSET_START {
            return buffercount + 1;
        }
        match buffer[buffercount] {
            ADD | SUB | MUL | DIV =>
                self.expression_length(buffer, self.expression_length(buffer, buffercount + 1)),
            _ => panic!("Unexpected token while calculating expression length")
        }
    }

    fn fitness_function(&mut self, prog: &[u8]) -> f64 {
        let mut fit: f64 = 0.0;
        let varnum = self.varnumber as usize;
        for i in 0..self.fitnesscases as usize {
            self.x[..varnum].clone_from_slice(&self.targets[i][..varnum]);
            self.pc = 0;
            let result = self.run(prog);
            fit += (result - self.targets[i][varnum]).abs();
        }
        -fit
    }

    fn random_operator(&mut self) -> u8 {
        self.rd.gen_range(0, FSET_END - FSET_START + 1) + FSET_START
    }

    fn random_x_index(&mut self) -> u8 {
        // 1/3 of the time...
        if self.rd.gen_range(0, 3) == 0 {
            // Return an index to an independent variable
            self.rd.gen_range(0, self.varnumber) as u8
        } else {
            // Return an index to an f64 constant
            self.rd.gen_range(self.varnumber, self.x.len() as u8)
        }
    }

    fn grow(&mut self, buffer: &mut Vec<u8>, pos: usize, depth: usize) -> isize {
        if pos >= MAX_LEN {
            return -1;
        }
        let mut prim: u8 = if pos == 0 { 1 } else { self.rd.gen_range(0, 2) };
        if prim == 0 || depth == 0 {
            prim = self.random_x_index();
            buffer[pos] = prim;
            pos as isize + 1
        }
        else  {
            prim = self.random_operator();
            match prim {
                ADD | SUB | MUL | DIV => {},
                _ => { return 0; }  // should never get here
            }
            buffer[pos] = prim;
            let one_child = self.grow(buffer, pos + 1, depth - 1);
            if one_child < 0 {
                return -1;
            }
            self.grow(buffer, one_child as usize, depth - 1)
        }
    }

    fn create_random_indiv(&mut self, depth: usize) -> Vec<u8> {
        let mut ind: Vec<u8> = vec![0; MAX_LEN];
        let mut len = self.grow(&mut ind, 0, depth);
        while len < 0 {
            len = self.grow(&mut ind, 0, depth);
        }
        ind.truncate(len as usize);
        ind.shrink_to_fit();
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

    // Returns 'best' competitor: less than 0 for left, 0 for same, greather than 0 for right.
    fn compare(&self, left: usize, right: usize) -> f64 {
        let fit_left = self.fitness[left];
        let fit_right = self.fitness[right];
        let average = (fit_left + fit_right) / 2.0;
        // Get a positive difference here if right has better fit than left
        let difference = fit_right - fit_left;
        let mut comparison_result = difference;
        let are_close = (difference / average).abs() * 100.0 <= CONSIDERED_CLOSE;
        if are_close {
            let len_left = self.pop[left].len();
            let len_right = self.pop[right].len();
            if len_left != len_right
                // Get a positive difference here if right is shorter than left
                { comparison_result = (len_left - len_right) as f64; }
        }
        comparison_result
    }

    fn tournament(&mut self, tsize: usize) -> usize {
        let mut best = self.rd.gen_range(0, POPSIZE);
        for _ in 1..tsize {
            let competitor = self.rd.gen_range(0, POPSIZE);
            if self.compare(best, competitor) > 0.0
                { best = competitor; }
        }
        best
    }

    fn negative_tournament(&mut self, tsize: usize) -> usize {
        let mut worst = self.rd.gen_range(0, POPSIZE);
        for _ in 1..tsize {
            let competitor = self.rd.gen_range(0, POPSIZE);
            if self.compare(worst, competitor) < 0.0
                { worst = competitor; }
        }
        worst
    }

    fn crossover(&mut self, parent1_index: usize, parent2_index: usize) -> Vec<u8> {
        let parent1 = &self.pop[parent1_index];
        let parent2 = &self.pop[parent2_index];
        let len1 = parent1.len();
        let len2 = parent2.len();
        let xo1start = self.rd.gen_range(0, len1);
        let xo1end = self.expression_length(&parent1, xo1start);
        let xo2start = self.rd.gen_range(0, len2);
        let xo2end = self.expression_length(&parent2, xo2start);
        let mut offspring = (&parent1[0..xo1start]).to_vec();
        offspring.extend((&parent2[xo2start..xo2end]).iter());
        offspring.extend((&parent1[xo1end..len1]).iter());
        let mut offspring_expr_len = self.expression_length(&offspring, 0);
        if offspring_expr_len > MAX_LEN {
            // It's too long. Truncate from the front.
            let overflow = offspring_expr_len - MAX_LEN;
            offspring = offspring[overflow..].to_vec();
            offspring_expr_len = self.expression_length(&offspring, 0);
        }
        offspring.truncate(offspring_expr_len as usize);
        offspring.shrink_to_fit();
        offspring
    }

    fn mutation(&mut self, parent_index: usize, pmut: f64) -> Vec<u8> {
        let parent = &self.pop[parent_index];
        let len = parent.len();
        let mut parentcopy = parent.clone();
        for i in 0..len {
            if self.rd.gen::<f64>() < pmut {
                let mutsite = i;
                if parentcopy[mutsite] < FSET_START {
                    parentcopy[mutsite] = self.random_x_index();
                }
                else {
                    match parentcopy[mutsite] {
                        ADD | SUB | MUL | DIV => {
                            parentcopy[mutsite] = self.random_operator();
                        },
                        _ => {}
                    }
                }
            }
        }
        parentcopy
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
                println!("Reached fitness better than 1e-5");
                self.print_parms();
                std::process::exit(0);
            }
            for _ in 0..POPSIZE {
                if self.rd.gen::<f64>() < CROSSOVER_PROB {
                    parent1 = self.tournament(TOURNAMENT_SIZE);
                    parent2 = self.tournament(TOURNAMENT_SIZE);
                    newind = self.crossover(parent1, parent2);
                } else {
                    parent = self.tournament(TOURNAMENT_SIZE);
                    newind = self.mutation(parent, PMUT_PER_NODE);
                }
                newfit = self.fitness_function(&newind);
                offspring = self.negative_tournament(TOURNAMENT_SIZE);
                self.pop[offspring] = newind;
                self.fitness[offspring] = newfit;
            }
            if gen % PRINT_FREQUENCY == 0 {
                self.stats(gen);
            }
        }
        println!("Reached maximum number of generations");
        self.print_parms();
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
            GENERATIONS={} \n\
            TSIZE={} \n\
            ----------------------------------",
            self.seed, MAX_LEN, POPSIZE, INITIAL_EXPRESSION_DEPTH, CROSSOVER_PROB, PMUT_PER_NODE,
            GENERATIONS, TOURNAMENT_SIZE
            );
    }

    fn print_indiv(&self, buffer: &[u8], buffercounter: usize) -> usize {
        let mut a1: usize = 0;
        if buffer[buffercounter] < FSET_START {
            print!("{}", &self.x_names[buffer[buffercounter] as usize]);
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

    fn stats(&mut self, gen: usize) {
        let mut best = self.rd.gen_range(0, POPSIZE);
        let mut best_size = 0;
        let mut node_count: usize = 0;
        self.fbestpop = self.fitness[best];
        self.favgpop = 0.0;
        for i in 0..POPSIZE {
            node_count += self.pop[i].len();
            self.favgpop += self.fitness[i];
            if self.fitness[i] > self.fbestpop {
                best = i;
                best_size = self.pop[i].len();
                self.fbestpop = self.fitness[i];
            }
        }
        let fpopsize = POPSIZE as f64;
        let avg_len = (node_count as f64) / fpopsize;
        self.favgpop /= fpopsize;
        print!("\
            {}secs \
            Generation={} \
            Avg Fitness={:.2} \
            Best Fitness={:.6} \
            Avg Size={:.2} \
            Best Size={} \n\
            Best Individual: ",
            self.starttime.elapsed().as_secs(),
            gen, -self.favgpop, -self.fbestpop, avg_len, best_size,
            );
        let buffer = &self.pop[best];
        self.print_indiv(buffer, 0);
        println!();
    }

}
