const range = 10

const example1 = () => {
  const range = 10
  let seed  = 5

  for (let i = 0; i < range; i++) {
    const result = ((seed * 8) % 11)
    seed = result
    console.log(result)
  }
}
// 7 1 8 9 6 4 10 3 2 5


const example2 = () => {
  const dateNow = Date.now()
  const playerID = 811182196
  const playerID2 = 333456712
  let randomSeed = ((dateNow + playerID) % 11)
  let seed = randomSeed

  console.log("Check Random Seed", seed)

  
  for (let i = 0; i < range; i++) {
    let result = ((seed * 8) % 11)

    if (result === 0) {
      console.log("Oh no its 0!")
      result = ((dateNow + playerID2) % 11) >> 1
    }

    seed = result
    console.log(result)
  }
}

example2()